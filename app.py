import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Rainfall Prediction App",
    page_icon="🌧️",
    layout="centered",
)

# -----------------------------
# Custom CSS for better UI
# -------------------
st.markdown("""
<style>
    .main {
    }
    h1 {
        color: #4a90e2;
        text-align: center;
        font-weight: bold;
    }
    .stSlider {
        margin-bottom: 35px; /* More spacing between sliders */
    }
    .stButton button {
        background-color: #4a90e2;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: 600;
        font-size: 1rem;
    }
    .stButton button:hover {
        background-color: #357ABD;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

with open("rainfall_prediction_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data['Model']
feature_names = model_data['Features_name']

RANGES = {
    "pressure": (985.0, 1050.0),      
    "dewpoint": (-10.0, 35.0),         
    "humidity": (0, 100),              
    "cloud": (0, 100),                
    "sunshine": (0.0, 15.0),           
    "winddirection": (0, 360),       
    "windspeed": (0.0, 75.0)           
}

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🌧️ Rainfall Prediction App")
st.markdown("### Enter today’s weather details to check if it might rain.")

pressure = st.slider("Pressure (hPa)", *RANGES["pressure"], value=1013.25, step=0.1)
dewpoint = st.slider("Dewpoint (°C)", *RANGES["dewpoint"], value=10.0, step=0.1)
humidity = st.slider("Humidity (%)", *RANGES["humidity"], value=50, step=1)
cloud1 = st.slider("Cloud Cover (%)", *RANGES["cloud"], value=50, step=1)
sunshine = st.slider("Sunshine (hours)", *RANGES["sunshine"], value=5.0, step=0.1)
winddirection = st.slider("Wind Direction (°)", *RANGES["winddirection"], value=180, step=5)
windspeed = st.slider("Wind Speed (km/h)", *RANGES["windspeed"], value=10.0, step=0.1)

if st.button("🔍 Predict Rainfall"):
    input_data = (pressure, dewpoint, humidity, cloud1, sunshine, winddirection, windspeed)
    input_df = pd.DataFrame([input_data], columns=feature_names)

    try:
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.markdown(
                '<div class="result-card" style="color:#d9534f; background:#ffecec;">☔ Rain is likely — Don’t forget your umbrella!</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-card" style="color:#5cb85c; background:#eaffea;">🌞 No rain expected — Enjoy your day!</div>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)
