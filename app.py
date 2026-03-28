import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
import random
import time
import plotly.graph_objects as go
import os
# ═══════════════════════════════════════════════════════════
# 🔹 PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Farming System",
    page_icon="🌱",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════
# 🔹 CONFIG
# ═══════════════════════════════════════════════════════════
BLYNK_TOKEN = os.getenv("BLYNK_TOKEN")
WEATHER_API = os.getenv("WEATHER_API")

# ═══════════════════════════════════════════════════════════
# 🔹 LOAD ML MODEL + ENCODERS + DATASET
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    model         = joblib.load("crop_model.pkl")
    label_encoder = joblib.load("label_encoder_model.pkl")
    scaler        = joblib.load("feature_scaler.pkl")
    season_enc    = joblib.load("season_encoder.pkl")
    state_enc     = joblib.load("state_encoder.pkl")
    soil_enc      = joblib.load("soil_encoder.pkl")
    return model, label_encoder, scaler, season_enc, state_enc, soil_enc

@st.cache_data
def load_data():
    df = pd.read_csv("datasets/slt_data.csv")
    df.columns      = df.columns.str.strip().str.replace(" ", "_")
    df["season"]    = df["season"].str.strip().str.capitalize()
    df["state"]     = df["state"].str.strip().str.title()
    df["label"]     = df["label"].str.strip().str.lower()
    df["soil_type"] = df["soil_type"].str.strip().str.lower()
    return df

model, label_encoder, scaler, season_enc, state_enc, soil_enc = load_models()
df = load_data()

# ═══════════════════════════════════════════════════════════
# 🔹 CROP MOISTURE PROFILES
# ═══════════════════════════════════════════════════════════
CROP_MOISTURE_PROFILES = {
    "rice"       : {"min": 70, "max": 100, "ideal": 85},
    "jute"       : {"min": 60, "max": 90,  "ideal": 75},
    "coconut"    : {"min": 55, "max": 80,  "ideal": 67},
    "banana"     : {"min": 55, "max": 80,  "ideal": 67},
    "maize"      : {"min": 45, "max": 70,  "ideal": 57},
    "cotton"     : {"min": 45, "max": 70,  "ideal": 57},
    "soybean"    : {"min": 45, "max": 70,  "ideal": 57},
    "tomato"     : {"min": 45, "max": 70,  "ideal": 57},
    "potato"     : {"min": 45, "max": 70,  "ideal": 57},
    "coffee"     : {"min": 45, "max": 70,  "ideal": 57},
    "papaya"     : {"min": 50, "max": 70,  "ideal": 60},
    "cabbage"    : {"min": 45, "max": 65,  "ideal": 55},
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
    "mustard"    : {"min": 20, "max": 40,  "ideal": 30},
    "mothbeans"  : {"min": 15, "max": 35,  "ideal": 25},
    "mungbean"   : {"min": 20, "max": 40,  "ideal": 30},
    "muskmelon"  : {"min": 20, "max": 40,  "ideal": 30},
    "watermelon" : {"min": 15, "max": 35,  "ideal": 25},
    "pomegranate": {"min": 15, "max": 35,  "ideal": 25},
    "grapes"     : {"min": 15, "max": 35,  "ideal": 25},
}
DEFAULT_PROFILE = {"min": 30, "max": 60, "ideal": 45}

# ═══════════════════════════════════════════════════════════
# 🔹 PSO PUMP OPTIMIZER
# ═══════════════════════════════════════════════════════════
def pso_pump_optimizer(crop_name, current_moisture, n_particles=30, iterations=60):
    profile    = CROP_MOISTURE_PROFILES.get(crop_name.lower(), DEFAULT_PROFILE)
    ideal, low = profile["ideal"], profile["min"]
    w, c1, c2  = 0.7, 1.5, 1.5
    particles  = np.random.uniform(0, 100, n_particles)
    velocities = np.random.uniform(-10, 10, n_particles)
    pbest      = particles.copy()
    pbest_cost = np.full(n_particles, np.inf)
    gbest      = 0.0
    gbest_cost = np.inf

    for _ in range(iterations):
        for i in range(n_particles):
            projected = min(current_moisture + particles[i] * 0.4, 100)
            diff      = projected - ideal
            cost      = abs(diff) + (0.5 * max(0, diff))
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest[i]      = particles[i]
            if cost < gbest_cost:
                gbest_cost = cost
                gbest      = particles[i]
        r1, r2     = np.random.rand(n_particles), np.random.rand(n_particles)
        velocities = np.clip(
            w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles),
            -20, 20
        )
        particles = np.clip(particles + velocities, 0, 100)

    pump_on       = current_moisture < low
    pump_duration = round(gbest, 1) if pump_on else 0.0
    moisture_gap  = round(low - current_moisture, 1) if pump_on else 0.0
    return pump_on, pump_duration, profile, moisture_gap

# ═══════════════════════════════════════════════════════════
# 🔹 HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════
def get_blynk(pin):
    try:
        r = requests.get(
            f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&{pin}",
            timeout=5
        )
        return float(r.text.strip())
    except Exception:
        return 0.0

def send_pump(value):
    try:
        requests.get(
            f"https://blynk.cloud/external/api/update?token={BLYNK_TOKEN}&V4={value}",
            timeout=5
        )
    except Exception:
        pass

def get_weather(city):
    try:
        url  = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API}&units=metric"
        data = requests.get(url, timeout=5).json()
        return data['main']['temp'], data['main']['humidity'], data.get('rain', {}).get('1h', 0)
    except Exception:
        return 30, 60, 0

def safe_encode(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else 0

def create_gauge(title, value, max_val, color):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 13}},
        gauge = {
            'axis' : {'range': [0, max_val]},
            'bar'  : {'color': color},
            'steps': [
                {'range': [0,              max_val * 0.33], 'color': '#2a2a2a'},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': '#333333'},
                {'range': [max_val * 0.66, max_val],        'color': '#3a3a3a'},
            ],
        }
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)', font_color='white'
    )
    return fig

def predict_crop_from_sensors(temp, humidity, N, P, K, ph, rainfall,
                               season="Kharif", state="Karnataka", soil="loamy"):
    s_code  = safe_encode(season_enc, season)
    st_code = safe_encode(state_enc,  state)
    sl_code = safe_encode(soil_enc,   soil)
    inp     = np.array([[N, P, K, temp, humidity, ph, rainfall, s_code, st_code, sl_code]])
    pred    = model.predict(scaler.transform(inp))
    return label_encoder.inverse_transform(pred)[0]

# ── Shared UI components ────────────────────────────────────────
def render_gauges(temp, humidity, soil):
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(create_gauge("🌡 Temperature (°C)", temp,     50,  "#e74c3c"),
                        width='stretch')
    with g2:
        st.plotly_chart(create_gauge("💧 Humidity (%)",     humidity, 100, "#3498db"),
                        width='stretch')
    with g3:
        st.plotly_chart(create_gauge("🌱 Soil Moisture (%)", soil,    100, "#2ecc71"),
                        width='stretch')

def render_weather(weather_temp, weather_hum, rain):
    st.subheader("🌦 Weather Conditions")
    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown(f"""
        <div style="background:#1e3c72;padding:18px;border-radius:12px;color:white;">
        🌡 <b>Temperature</b><br><span style="font-size:24px">{weather_temp:.1f}°C</span>
        </div>""", unsafe_allow_html=True)
    with wc2:
        st.markdown(f"""
        <div style="background:#2a5298;padding:18px;border-radius:12px;color:white;">
        💧 <b>Humidity</b><br><span style="font-size:24px">{weather_hum}%</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if rain > 0:
        st.markdown(f"""
        <div style="background:#e74c3c;padding:14px;border-radius:12px;
                    color:white;font-weight:bold;text-align:center;">
        🌧 Rain Expected ({rain} mm/h) — Irrigation automatically paused
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#27ae60;padding:14px;border-radius:12px;
                    color:white;font-weight:bold;text-align:center;">
        ☀ No Rain Expected — Irrigation system active
        </div>""", unsafe_allow_html=True)

def render_crop_profile_card(crop_name):
    profile = CROP_MOISTURE_PROFILES.get(crop_name.lower(), DEFAULT_PROFILE)
    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid #444;padding:16px;
                border-radius:12px;color:white;margin-bottom:10px;">
        <b style="font-size:16px;">🌾 {crop_name.title()} — Moisture Requirements</b><br><br>
        🔴 &nbsp;Min &nbsp;: <b>{profile['min']}%</b> &nbsp;&nbsp;
        🟢 Ideal : <b>{profile['ideal']}%</b> &nbsp;&nbsp;
        🔵 Max &nbsp;: <b>{profile['max']}%</b>
    </div>""", unsafe_allow_html=True)

def render_pump_section(crop_name, soil_moisture, rain):
    pump_on, pump_duration, profile, moisture_gap = pso_pump_optimizer(crop_name, soil_moisture)
    if rain > 0:
        pump_on, pump_duration = False, 0.0
    send_pump(1 if pump_on else 0)

    if rain > 0:
        p_color, p_msg = "#3498db", "🌧 Pump OFF — Rain detected, irrigation paused"
    elif pump_on:
        p_color, p_msg = "#00ff00", f"🚰 Pump ON — PSO Optimal Duration: {pump_duration}%"
    else:
        p_color, p_msg = "#27ae60", "✅ Pump OFF — Soil moisture is optimal"

    if soil_moisture < profile["min"]:
        m_color = "#c0392b"
        m_msg   = f"🔴 Too Dry! Needs >{profile['min']}% for {crop_name.title()}"
    elif soil_moisture > profile["max"]:
        m_color = "#2980b9"
        m_msg   = f"🔵 Too Wet! Reduce below {profile['max']}% for {crop_name.title()}"
    elif abs(soil_moisture - profile["ideal"]) <= 5:
        m_color = "#27ae60"
        m_msg   = f"🟢 Optimal moisture for {crop_name.title()}!"
    else:
        m_color = "#d35400"
        m_msg   = f"🟡 Acceptable — Ideal is {profile['ideal']}% for {crop_name.title()}"

    st.markdown(f"""
    <div style="background:{p_color};padding:18px;border-radius:12px;
                color:white;font-weight:bold;font-size:18px;
                text-align:center;margin-bottom:10px;">{p_msg}</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{m_color};padding:14px;border-radius:12px;
                color:white;font-weight:bold;text-align:center;
                margin-bottom:10px;">{m_msg}</div>
    """, unsafe_allow_html=True)

    with st.expander("🤖 PSO Optimization Details"):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Current Moisture", f"{soil_moisture:.1f}%")
        p2.metric("Min Threshold",    f"{profile['min']}%")
        p3.metric("Ideal Target",     f"{profile['ideal']}%")
        p4.metric("PSO Duration",     f"{pump_duration}%")
        st.caption(
            "PSO runs 30 particles × 60 iterations to find the optimal pump "
            "duration that brings soil moisture to the ideal level without over-watering."
        )

# ═══════════════════════════════════════════════════════════
# 🔹 SIDEBAR
# ═══════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/color/96/plant-under-rain.png", width=70)
st.sidebar.title("🌱 Smart Farming")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🌾 Crop Recommendation"],
    label_visibility="collapsed"
)

#st.sidebar.markdown("---")
weather_city ="Karnataka"
st.sidebar.markdown("---")
st.sidebar.caption("AI Smart Farming ")


# ═══════════════════════════════════════════════════════════════════
# 📊 PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.title("📊 Smart Farm Dashboard")
    st.markdown("Monitor sensors, weather, and smart irrigation in real time.")
    st.markdown("---")

    # Fetch weather once — shared by both tabs
    weather_temp, weather_hum, rain = get_weather(weather_city)

    tab1, tab2 = st.tabs(["📈 Data Values", "🔌 ESP32"])

    # ══════════════════════════════════════════════════════
    # TAB 1: DATA VALUES
    # Fully automated — ML predicts crop, PSO controls pump
    # No user input required at all
    # ══════════════════════════════════════════════════════
    with tab1:
        st.markdown("##### 🤖 Fully Automated ")
        st.markdown("---")

        # Random sensor values (simulated data)
        temp     = round(random.uniform(20, 35), 1)
        humidity = round(random.uniform(40, 80),  1)
        soil     = round(random.uniform(10, 90),  1)
        N        = random.randint(0, 140)
        P        = random.randint(5, 145)
        K        = random.randint(5, 205)
        ph       = round(random.uniform(5.5, 7.5), 2)
        rainfall = random.randint(20, 200)

        # ML auto-predicts crop from sensor data
        predicted_crop = predict_crop_from_sensors(temp, humidity, N, P, K, ph, rainfall)

        # ── Sensor Gauges ────────────────────────────────────
        st.subheader("🌡 Sensor Readings")
        render_gauges(temp, humidity, soil)
        st.markdown("---")

        # ── ML Predicted Crop ────────────────────────────────
        st.subheader("🌾 ML Predicted Crop")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1f4037,#99f2c8);
                    padding:18px;border-radius:12px;color:black;
                    font-weight:bold;font-size:22px;text-align:center;">
            🌾 {predicted_crop.title()}
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        render_crop_profile_card(predicted_crop)
        st.markdown("---")

        # ── PSO Pump (fully automatic) ───────────────────────
        st.subheader("🚰 Smart Pump Control (PSO)")
        render_pump_section(predicted_crop, soil, rain)
        st.markdown("---")

        # ── Weather ──────────────────────────────────────────
        render_weather(weather_temp, weather_hum, rain)

    # ══════════════════════════════════════════════════════
    # TAB 2: ESP32
    # User selects crop. Random values.
    # Just swap 3 lines when hardware is ready.
    # ══════════════════════════════════════════════════════
    with tab2:
        st.markdown("##### 🔌 ESP32 Mode — Select your crop. .")
        st.markdown("---")

        # ── User selects crop ────────────────────────────────
        st.subheader("🌾 Select Your Crop")
        selected_crop = st.selectbox(
            "Choose the crop planted in your field:",
            sorted(CROP_MOISTURE_PROFILES.keys()),
            format_func=lambda x: x.title()
        )
        st.markdown("---")

       
        # ⬇ When ESP32 is connected, replace these 3 lines:
        # temp     = get_blynk("V0")
        # humidity = get_blynk("V1")
        # soil     = get_blynk("V2")
        temp     = round(random.uniform(20, 35), 1)
        humidity = round(random.uniform(40, 80),  1)
        soil     = round(random.uniform(10, 90),  1)

        # ── Sensor Gauges ────────────────────────────────────
        st.subheader("🌡 Sensor Readings")
        render_gauges(temp, humidity, soil)
        st.markdown("---")

        # ── Selected Crop Profile ────────────────────────────
        st.subheader(f"🌾 {selected_crop.title()} — Moisture Profile")
        render_crop_profile_card(selected_crop)
        st.markdown("---")

        # ── PSO Pump (user crop) ─────────────────────────────
        st.subheader("🚰 Smart Pump Control (PSO)")
        render_pump_section(selected_crop, soil, rain)
        st.markdown("---")

        # ── Weather ──────────────────────────────────────────
        render_weather(weather_temp, weather_hum, rain)

        st.markdown("---")
        st.info(
            "🔌 **Ready for ESP32** — When your hardware is connected, "
            "replace the 3 random value lines in Tab 2 with "
            "`get_blynk('V0')`, `get_blynk('V1')`, `get_blynk('V2')`."
        )

    # ── Auto Refresh ─────────────────────────────────────────────
    time.sleep(10)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════
# 🌾 PAGE 2 — CROP RECOMMENDATION
# No weather section. No rain alert.
# ═══════════════════════════════════════════════════════════════════
elif page == "🌾 Crop Recommendation":

    language = st.sidebar.selectbox("🌐 Language", ["English", "Kannada/ಕನ್ನಡ", "Tamil/தமிழ்", "Hindi/हिन्दी", "Marathi/मराठी", "Punjabi/ਪੰਜਾਬੀ"])

    translations = {
        "English": {
            "title"   : "🌱 Smart Farming Decision System",
            "state"   : "Select State",
            "season"  : "Select Season",
            "soil"    : "Select Soil Type",
            "button"  : "Get Recommendation",
            "top"     : "🌟 Top Recommended Crops",
            "analysis": "📊 Crop Analysis",
            "all"     : "📋 All Suitable Crops",
            "no_crops": "No crops found for selected state and season.",
        },
        "Kannada/ಕನ್ನಡ": {
            "title"   : "🌱 ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ವ್ಯವಸ್ಥೆ",
            "state"   : "ರಾಜ್ಯ ಆಯ್ಕೆಮಾಡಿ",
            "season"  : "ಋತು ಆಯ್ಕೆಮಾಡಿ",
            "soil"    : "ಮಣ್ಣಿನ ಪ್ರಕಾರ ಆಯ್ಕೆಮಾಡಿ",
            "button"  : "ಶಿಫಾರಸು ಪಡೆಯಿರಿ",
            "top"     : "🌟 ಅತ್ಯುತ್ತಮ ಬೆಳೆಗಳು",
            "analysis": "📊 ಬೆಳೆ ವಿಶ್ಲೇಷಣೆ",
            "all"     : "📋 ಎಲ್ಲಾ ಬೆಳೆಗಳು",
            "no_crops": "ಆಯ್ಕೆ ಮಾಡಿದ ರಾಜ್ಯ ಮತ್ತು ಋತುವಿಗೆ ಯಾವುದೇ ಬೆಳೆ ಸಿಗಲಿಲ್ಲ.",
        },
        "Tamil/தமிழ்": {
           "title"   : "🌱 நுண்ணிய வேளாண்மை முறை",
           "state"   : "மாநிலத்தை தேர்ந்தெடுக்கவும்",
           "season"  : "பருவகாலத்தை தேர்ந்தெடுக்கவும்",
           "soil"    : "மண் வகையை தேர்ந்தெடுக்கவும்",
           "button"  : "பரிந்துரையை பெறுக",
           "top"     : "🌟 சிறந்த பயிர்கள்",
           "analysis": "📊 பயிர் பகுப்பாய்வு",
           "all"     : "📋 அனைத்து பயிர்களும்",
           "no_crops": "தேர்ந்தெடுத்த மாநிலம் மற்றும் பருவகாலத்திற்கு பயிர்கள் எதுவும் இல்லை.",
        },
        "Hindi/हिन्दी": {
            "title"   : "🌱 स्मार्ट कृषि प्रणाली",
            "state"   : "राज्य चुनें",
            "season"  : "मौसम चुनें",
            "soil"    : "मिट्टी का प्रकार चुनें",
            "button"  : "सुझाव प्राप्त करें",
            "top"     : "🌟 शीर्ष फसलें",
            "analysis": "📊 फसल विश्लेषण",
            "all"     : "📋 सभी फसलें",
            "no_crops": "चुने गए राज्य और मौसम के लिए कोई फसल नहीं मिली।",
        },
        "Marathi/मराठी": {
            "title"   : "🌱 स्मार्ट शेती निर्णय प्रणाली",
            "state"   : "राज्य निवडा",
            "season"  : "हंगाम निवडा",
            "soil"    : "मातीचा प्रकार निवडा",
            "button"  : "शिफारस मिळवा",
            "top"     : "🌟 शीर्ष शिफारस केलेली पिके",
            "analysis": "📊 पीक विश्लेषण",
            "all"     : "📋 सर्व योग्य पिके",
            "no_crops": "निवडलेल्या राज्य आणि हंगामासाठी कोणतेही पीक आढळले नाही.",
        },
        "Punjabi/ਪੰਜਾਬੀ": {
            "title"   : "🌱 ਸਮਾਰਟ ਖੇਤੀ ਫੈਸਲਾ ਪ੍ਰਣਾਲੀ",
            "state"   : "ਰਾਜ ਚੁਣੋ",
            "season"  : "ਮੌਸਮ ਚੁਣੋ",
            "soil"    : "ਮਿੱਟੀ ਦੀ ਕਿਸਮ ਚੁਣੋ",
            "button"  : "ਸਿਫ਼ਾਰਸ਼ ਪ੍ਰਾਪਤ ਕਰੋ",
            "top"     : "🌟 ਚੋਟੀ ਦੀਆਂ ਫ਼ਸਲਾਂ",
            "analysis": "📊 ਫ਼ਸਲ ਵਿਸ਼ਲੇਸ਼ਣ",
            "all"     : "📋 ਸਾਰੀਆਂ ਢੁਕਵੀਆਂ ਫ਼ਸਲਾਂ",
            "no_crops": "ਚੁਣੇ ਗਏ ਰਾਜ ਅਤੇ ਮੌਸਮ ਲਈ ਕੋਈ ਫ਼ਸਲ ਨਹੀਂ ਮਿਲੀ।",
       },
    }
    t = translations[language]

    st.title(t["title"])
    st.markdown("Select your location and soil details to get AI-powered crop suggestions.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        state  = st.selectbox(t["state"],  sorted(df["state"].unique()))
    with c2:
        season = st.selectbox(t["season"], sorted(df["season"].unique()))
    with c3:
        soil_sel = st.selectbox(t["soil"], sorted(df["soil_type"].unique()))

    st.markdown("")

    if st.button(t["button"], type="primary", use_container_width=True):

        # Weather used only for ML input features, not displayed
        temp, humidity, rain = get_weather(state)

        filtered = df[
            (df["state"]  == state) &
            (df["season"] == season)
        ]

        if filtered.empty:
            st.warning(t["no_crops"])
        else:
            unique_crops = filtered["label"].unique()
            crop_scores  = []

            s_code  = safe_encode(season_enc, season)
            st_code = safe_encode(state_enc,  state)
            sl_code = safe_encode(soil_enc,   soil_sel)

            for crop_name in unique_crops:
                crop_rows = filtered[filtered["label"] == crop_name]
                avg_row   = crop_rows[["N", "P", "K", "ph", "rainfall"]].mean()

                input_data = np.array([[
                    avg_row["N"], avg_row["P"], avg_row["K"],
                    temp, humidity,
                    avg_row["ph"],
                    rain if rain > 0 else avg_row["rainfall"],
                    s_code, st_code, sl_code
                ]])
                input_scaled = scaler.transform(input_data)

                if crop_name in label_encoder.classes_:
                    probs      = model.predict_proba(input_scaled)[0]
                    crop_index = label_encoder.transform([crop_name])[0]
                    score      = probs[crop_index]
                else:
                    score = 0.01

                crop_scores.append((crop_name, score))

            crop_scores.sort(key=lambda x: x[1], reverse=True)

            # ── Top 3 side-by-side cards ─────────────────────
            st.subheader(t["top"])
            medals = ["🥇", "🥈", "🥉"]
            m1, m2, m3 = st.columns(3)
            for i, col in enumerate([m1, m2, m3]):
                if i < len(crop_scores):
                    crop, score = crop_scores[i]
                    with col:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#1f4037,#99f2c8);
                                    padding:22px;border-radius:14px;text-align:center;
                                    color:black;font-weight:bold;font-size:17px;min-height:100px;">
                            {medals[i]}<br>{crop.title()}<br>
                            <span style="font-size:24px">{score*100:.1f}%</span>
                        </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Bar Chart ────────────────────────────────────
            st.subheader(t["analysis"])
            chart_data = pd.DataFrame({
                "Crop" : [c[0].title() for c in crop_scores],
                "Score": [round(c[1] * 100, 2) for c in crop_scores]
            })
            st.bar_chart(chart_data.set_index("Crop"))
            st.markdown("---")

            # ── All Crops ────────────────────────────────────
            st.subheader(t["all"])
            for crop, score in crop_scores:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"🌾 {crop.title()}")
                with col2:
                    st.write(f"{score * 100:.1f}%")
                st.progress(float(min(score, 1.0)))
