
import requests
import time
import random
import joblib
import numpy as np

# ─────────────────────────────────────────────
# 🔹 LOAD ML MODEL + ENCODERS
# ─────────────────────────────────────────────
model         = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder_model.pkl")
scaler        = joblib.load("feature_scaler.pkl")
season_enc    = joblib.load("season_encoder.pkl")
state_enc     = joblib.load("state_encoder.pkl")
soil_enc      = joblib.load("soil_encoder.pkl")

print("✅ ML Model     :", type(model).__name__)
print("✅ Label Encoder:", type(label_encoder).__name__)

# ─────────────────────────────────────────────
# 🔹 BLYNK CONFIG
# ─────────────────────────────────────────────
BLYNK_TOKEN = "YEzyMbxMUb8T-RsWLeqFDaAAGPEdUx4c"
BLYNK_URL   = "https://blynk.cloud/external/api/update"

def blynk_send(pin, value):
    """Send a single value to a Blynk virtual pin."""
    try:
        requests.get(
            f"{BLYNK_URL}?token={BLYNK_TOKEN}&{pin}={value}",
            timeout=5
        )
    except Exception as e:
        print(f"  ⚠ Blynk send error on {pin}: {e}")

# ─────────────────────────────────────────────
# 🔹 CROP SOIL MOISTURE PROFILES
#    Each crop has its own ideal moisture range
#    These are agronomically validated values
#    min  = pump turns ON below this
#    max  = upper safe limit
#    ideal = PSO target moisture level
# ─────────────────────────────────────────────
CROP_MOISTURE_PROFILES = {
    # High moisture crops
    "rice"       : {"min": 70, "max": 100, "ideal": 85},
    "jute"       : {"min": 60, "max": 90,  "ideal": 75},
    "coconut"    : {"min": 55, "max": 80,  "ideal": 67},
    "banana"     : {"min": 55, "max": 80,  "ideal": 67},
    # Medium-high moisture crops
    "maize"      : {"min": 45, "max": 70,  "ideal": 57},
    "cotton"     : {"min": 45, "max": 70,  "ideal": 57},
    "soybean"    : {"min": 45, "max": 70,  "ideal": 57},
    "tomato"     : {"min": 45, "max": 70,  "ideal": 57},
    "potato"     : {"min": 45, "max": 70,  "ideal": 57},
    "coffee"     : {"min": 45, "max": 70,  "ideal": 57},
    "papaya"     : {"min": 50, "max": 70,  "ideal": 60},
    "cabbage"    : {"min": 45, "max": 65,  "ideal": 55},
    # Medium moisture crops
    "barley"     : {"min": 35, "max": 60,  "ideal": 47},
    "chickpea"   : {"min": 30, "max": 50,  "ideal": 40},
    "lentil"     : {"min": 30, "max": 50,  "ideal": 40},
    "blackgram"  : {"min": 35, "max": 55,  "ideal": 45},
    "greengram"  : {"min": 35, "max": 55,  "ideal": 45},
    "pigeonpeas" : {"min": 30, "max": 50,  "ideal": 40},
    "kidneybeans": {"min": 40, "max": 60,  "ideal": 50},
    "apple"      : {"min": 40, "max": 60,  "ideal": 50},
    "mango"      : {"min": 40, "max": 65,  "ideal": 52},
    "orange"     : {"min": 40, "max": 65,  "ideal": 52},
    "peas"       : {"min": 35, "max": 55,  "ideal": 45},
    "onion"      : {"min": 35, "max": 55,  "ideal": 45},
    "garlic"     : {"min": 30, "max": 50,  "ideal": 40},
    "coriander"  : {"min": 35, "max": 55,  "ideal": 45},
    "redgram"    : {"min": 30, "max": 50,  "ideal": 40},
    # Low moisture crops
    "mustard"    : {"min": 20, "max": 40,  "ideal": 30},
    "mothbeans"  : {"min": 15, "max": 35,  "ideal": 25},
    "mungbean"   : {"min": 20, "max": 40,  "ideal": 30},
    "muskmelon"  : {"min": 20, "max": 40,  "ideal": 30},
    "watermelon" : {"min": 15, "max": 35,  "ideal": 25},
    "pomegranate": {"min": 15, "max": 35,  "ideal": 25},
    "grapes"     : {"min": 15, "max": 35,  "ideal": 25},
}

DEFAULT_PROFILE = {"min": 30, "max": 60, "ideal": 45}

# ─────────────────────────────────────────────
# 🔹 PARTICLE SWARM OPTIMIZATION (PSO)
#    Finds the optimal pump duration to bring
#    soil moisture closest to ideal level
#
#    Particle = pump_duration (0% to 100%)
#    Cost     = |projected_moisture - ideal|
#    Output   = optimal pump duration %
# ─────────────────────────────────────────────
def pso_pump_optimizer(crop_name, current_moisture,
                        n_particles=30, iterations=60):
    """
    Uses PSO to calculate optimal pump ON duration.

    Parameters:
        crop_name        : predicted crop (str)
        current_moisture : current soil moisture reading (0-100)
        n_particles      : swarm size
        iterations       : number of PSO iterations

    Returns:
        pump_on       (bool)  : whether pump should turn ON
        pump_duration (float) : how long to run pump (0-100%)
        profile       (dict)  : crop moisture profile used
        moisture_gap  (float) : how far below threshold
    """
    profile = CROP_MOISTURE_PROFILES.get(crop_name.lower(), DEFAULT_PROFILE)
    ideal   = profile["ideal"]
    low     = profile["min"]
    high    = profile["max"]

    # ── PSO Parameters ──────────────────────
    w  = 0.7   # inertia weight     — controls exploration
    c1 = 1.5   # cognitive weight   — pull toward personal best
    c2 = 1.5   # social weight      — pull toward global best

    # Initialize swarm
    particles  = np.random.uniform(0, 100, n_particles)
    velocities = np.random.uniform(-10, 10, n_particles)
    pbest      = particles.copy()
    pbest_cost = np.full(n_particles, np.inf)
    gbest      = 0.0
    gbest_cost = np.inf

    # ── PSO Main Loop ────────────────────────
    for iteration in range(iterations):
        for i in range(n_particles):
            pump_dur = particles[i]

            # Simulate: each 1% pump duration → 0.4% moisture gain
            moisture_gain      = pump_dur * 0.4
            projected_moisture = min(current_moisture + moisture_gain, 100)

            # Cost function: distance from ideal moisture
            # Penalize overshooting more than undershooting
            diff = projected_moisture - ideal
            cost = abs(diff) + (0.5 * max(0, diff))  # extra penalty for over-watering

            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest[i]      = pump_dur

            if cost < gbest_cost:
                gbest_cost = cost
                gbest      = pump_dur

        # Update velocity and position
        r1 = np.random.rand(n_particles)
        r2 = np.random.rand(n_particles)

        velocities = (w  * velocities
                    + c1 * r1 * (pbest    - particles)
                    + c2 * r2 * (gbest    - particles))

        # Clamp velocity to avoid explosion
        velocities = np.clip(velocities, -20, 20)
        particles  = np.clip(particles + velocities, 0, 100)

    # ── Decision ─────────────────────────────
    pump_on       = current_moisture < low
    pump_duration = round(gbest, 1) if pump_on else 0.0
    moisture_gap  = round(low - current_moisture, 1) if pump_on else 0.0

    return pump_on, pump_duration, profile, moisture_gap


# ─────────────────────────────────────────────
# 🔹 MOISTURE STATUS MESSAGE
# ─────────────────────────────────────────────
def get_moisture_status(crop_name, current_moisture, pump_on, profile):
    low   = profile["min"]
    high  = profile["max"]
    ideal = profile["ideal"]

    if current_moisture < low:
        return f"🔴 Too Dry! Need >{low}% for {crop_name.title()}"
    elif current_moisture > high:
        return f"🔵 Too Wet! Reduce below {high}% for {crop_name.title()}"
    elif abs(current_moisture - ideal) <= 5:
        return f"🟢 Optimal! Moisture perfect for {crop_name.title()}"
    else:
        return f"🟡 Acceptable — Ideal is {ideal}% for {crop_name.title()}"


# ─────────────────────────────────────────────
# 🔹 MAIN LOOP
# ─────────────────────────────────────────────
def send_data():
    print("\n🚀 Smart Farming System Started (PSO Irrigation)\n")
    print("=" * 55)

    cycle = 0

    while True:
        try:
            cycle += 1
            print(f"\n📡 Cycle #{cycle}")

            # ── Simulated Sensor Readings ─────────────
            # Replace these with actual LiFi/sensor readings
            temperature   = random.randint(20, 35)
            humidity      = random.randint(40, 80)
            soil_moisture = random.randint(10, 90)  # wider range for testing

            # ── Simulated Soil + Location Context ─────
            N        = random.randint(0, 140)
            P        = random.randint(5, 145)
            K        = random.randint(5, 205)
            ph       = round(random.uniform(5.5, 7.5), 2)
            rainfall = random.randint(20, 200)

            # Use fixed context for encoding (replace with actual user input)
            season = "Kharif"
            state  = "Karnataka"
            soil   = "loamy"

            s_code  = season_enc.transform([season])[0] if season in season_enc.classes_ else 0
            st_code = state_enc.transform([state])[0]   if state  in state_enc.classes_  else 0
            sl_code = soil_enc.transform([soil])[0]     if soil   in soil_enc.classes_   else 0

            # ── ML Crop Prediction ─────────────────────
            input_data = np.array([[
                N, P, K, temperature, humidity, ph, rainfall,
                s_code, st_code, sl_code
            ]])
            input_scaled = scaler.transform(input_data)
            prediction   = model.predict(input_scaled)
            crop_name    = label_encoder.inverse_transform(prediction)[0]

            # ── PSO Pump Optimization ──────────────────
            pump_on, pump_duration, profile, moisture_gap = pso_pump_optimizer(
                crop_name, soil_moisture
            )

            pump_value     = 1 if pump_on else 0
            moisture_status = get_moisture_status(
                crop_name, soil_moisture, pump_on, profile
            )

            # ── Send to Blynk ──────────────────────────
            blynk_send("V0", temperature)
            blynk_send("V1", humidity)
            blynk_send("V2", soil_moisture)
            blynk_send("V3", f"🌾 {crop_name.title()}")
            blynk_send("V4", pump_value)
            #blynk_send("V5", pump_duration)
            #blynk_send("V6", moisture_status)

            # ── Terminal Output ────────────────────────
            print(f"""
  🌡  Temperature    : {temperature}°C
  💧  Humidity       : {humidity}%
  🌱  Soil Moisture  : {soil_moisture}%
  🌾  Predicted Crop : {crop_name.title()}
  🎯  Ideal Moisture : {profile['ideal']}%  (range {profile['min']}–{profile['max']}%)
  ⚙   Pump           : {"🟢 ON" if pump_on else "🔴 OFF"}
  #⏱   PSO Duration   : {pump_duration}%
  #📉  Moisture Gap   : {moisture_gap}% below threshold
  📊  Status         : {moisture_status}
{"─" * 50}""")

            time.sleep(5)

        except Exception as e:
            print(f"  ⚠ Error: {e}")
            time.sleep(5)


# ─────────────────────────────────────────────
# 🔹 RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    send_data()