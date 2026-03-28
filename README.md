🌱 AI-Based Smart Farming Using LiFi
     
An intelligent smart farming system that combines Machine Learning, Particle Swarm Optimization (PSO), IoT sensors (ESP32), and LiFi communication to recommend crops and automate irrigation based on real-time soil and weather conditions.
________________________________________
📌 Table of Contents
•	Overview
•	Features
•	System Architecture
•	Tech Stack
•	Dataset
•	ML Model
•	PSO Irrigation
•	Installation
•	Usage
•	Project Structure
•	Screenshots
•	Future Work
•	Team
________________________________________
📖 Overview
Traditional farming relies on manual judgment for crop selection and irrigation, leading to inefficiency and crop loss. This project addresses these challenges by building an end-to-end AI-powered smart farming platform that:
•	Recommends the best crops based on soil, season, state, and weather
•	Automates pump control using PSO optimization based on crop-specific moisture needs
•	Transmits sensor data wirelessly via LiFi (Light Fidelity) technology
•	Provides a real-time dashboard with multilingual support
________________________________________
✨ Features
📊 Smart Dashboard
•	Real-time temperature, humidity, and soil moisture gauges
•	Two modes — Data Values (fully automated) and ESP32 (user-controlled)
•	Live weather conditions and rain prediction via OpenWeatherMap API
•	Auto-refresh every 5 seconds
🌾 Crop Recommendation System
•	AI-powered crop suggestions based on state, season, and soil type
•	Top 3 crops displayed with confidence scores
•	Full crop analysis bar chart and ranked list
•	Supports 6 languages — English, Kannada, Hindi, Tamil, Marathi, Punjabi
🚰 Smart Irrigation (PSO)
•	Particle Swarm Optimization calculates optimal pump ON/OFF duration
•	Crop-specific soil moisture profiles (34 crops)
•	Rain detection automatically pauses irrigation
•	Sends pump control signal to ESP32 via Blynk IoT
🔌 IoT Integration
•	Compatible with ESP32 microcontroller
•	Blynk IoT platform for sensor data transmission
•	Ready for LiFi module integration for wireless data transfer
________________________________________
🏗 System Architecture
Sensors (Temp, Humidity, Soil Moisture)
        ↓
    ESP32 Module
        ↓ (LiFi / Blynk)
    Python Backend
     ├── ML Model (Random Forest) → Crop Prediction
     ├── PSO Optimizer → Pump Duration
     └── Weather API → Rain Detection
        ↓
  Streamlit Dashboard
     ├── 📊 Dashboard Page (Live Sensor Data)
     └── 🌾 Crop Recommendation Page
        ↓
    Pump Control (via Blynk V4)
________________________________________
🛠 Tech Stack
Category	Technology
Frontend	Streamlit, Plotly
Backend	Python 3.10+
ML Model	Random Forest (scikit-learn)
Optimization	Particle Swarm Optimization (PSO)
IoT Platform	ESP32, Blynk IoT
Communication	LiFi (Light Fidelity)
Weather API	OpenWeatherMap
Data	Pandas, NumPy
Model Saving	Joblib
________________________________________
📂 Dataset
The project uses slt_data.csv — a curated agricultural dataset containing:
Feature	Description
N, P, K	Soil nutrient levels
temperature	Ambient temperature (°C)
humidity	Relative humidity (%)
PH	Soil pH value
rainfall	Rainfall (mm)
season	Kharif / Rabi / Perennial
soil type	Sandy / Black / Loamy / Red / Clay
state	Karnataka / Punjab / Tamil Nadu / Maharashtra / Uttar Pradesh
label	Crop name (34 crops)
Total records: 3,400 | Crops: 34 | Balanced: 100 samples per crop
________________________________________
🤖 ML Model
•	Algorithm: Random Forest Classifier
•	Features: N, P, K, temperature, humidity, pH, rainfall, season, state, soil type (10 features)
•	Test Accuracy: ~95%
•	Cross-validation: ~93.88%
•	Classes: 34 crop types
Training the Model
python train_best_model.py
This will:
1.	Train and benchmark 8 ML models
2.	Auto-select the best performing model
3.	Save all .pkl files automatically
________________________________________
🔬 PSO Irrigation Logic
Particle Swarm Optimization is used to find the optimal pump duration to bring soil moisture to the ideal level for each crop.
Particles  = candidate pump durations (0–100%)
Cost       = |projected_moisture - ideal_moisture|
Iterations = 60 rounds × 30 particles
Output     = optimal pump ON duration (%)
Each crop has a defined moisture profile:
"rice"    : {"min": 70, "max": 100, "ideal": 85}
"mustard" : {"min": 20, "max": 40,  "ideal": 30}
"grapes"  : {"min": 15, "max": 35,  "ideal": 25}
# ... 34 crops total
________________________________________
⚙️ Installation
1. Clone the Repository
git clone https://github.com/YourUsername/smart-farming-lifi.git
cd smart-farming-lifi
2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
3. Install Dependencies
pip install -r requirements.txt
4. Train the Model
python train_best_model.py
5. Run the Dashboard
streamlit run app_final.py
________________________________________
🚀 Usage
Dashboard — Data Values Mode
•	Fully automated
•	Random sensor values → ML predicts crop → PSO controls pump
•	No user input required
Dashboard — ESP32 Mode
•	Select your crop from the dropdown
•	System uses PSO to decide pump ON/OFF based on soil moisture
•	Replace random values with get_blynk("V0/V1/V2") when ESP32 is connected
Crop Recommendation
•	Select State, Season, and Soil Type
•	Click Get Recommendation
•	View top 3 crops with confidence scores, bar chart, and full ranked list
________________________________________
📁 Project Structure
smart-farming-lifi/
│
├── app_final.py                        # Main Streamlit dashboard
├── train_best_model.py                 # Model training script
├── blynk_pso_smart_irrigation.py       # Blynk + PSO backend script
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
├── datasets/
│   └── slt_data.csv                    # Training dataset
│
├── crop_model.pkl                      # Trained ML model
├── label_encoder_model.pkl             # Crop label encoder
├── feature_scaler.pkl                  # Feature scaler
├── season_encoder.pkl                  # Season encoder
├── state_encoder.pkl                   # State encoder
└── soil_encoder.pkl                    # Soil type encoder
________________________________________
🔮 Future Work
•	[ ] Integrate real LiFi hardware module for wireless sensor data
•	[ ] Connect live ESP32 sensors (NPK, pH, soil moisture)
•	[ ] Add yield prediction as a second ML model
•	[ ] Expand dataset to include more Indian states and crops
•	[ ] Mobile app integration via Flutter
•	[ ] Add SMS/email alerts when soil moisture is critically low
•	[ ] Historical data logging and trend analysis charts
________________________________________
👨‍💻 Team
Name	Role
Mohammed Shadab	 ML Engineer / Full Stack
	IoT Integration
	Data Collection
Project developed as part of Capstone Project 2026 Department of Electronics & communication  Engineering
________________________________________
📄 License
This project is licensed under the MIT License — feel free to use and modify with attribution.
________________________________________
🙏 Acknowledgements
•	Streamlit — for the amazing dashboard framework
•	Blynk IoT — for IoT platform support
•	OpenWeatherMap — for weather API
•	scikit-learn — for ML tools
•	Anthropic Claude — for development assistance
________________________________________
Made with ❤️ for smarter, sustainable farming 🌾

