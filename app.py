import streamlit as st
import joblib
import numpy as np

st.title("🎓 Student Focus Level Predictor")

st.write(
    "Predict whether a student has High, Medium or Low focus level."
)

st.info("Model Accuracy: 96.2% (Random Forest)")

st.markdown("---")

model = joblib.load("focus_level_model.pkl")
scaler = joblib.load("scaler.pkl")  


login_frequency = st.number_input("Login Frequency", min_value=1)
session_duration = st.number_input("Session Duration")
video_watch_time = st.number_input("Video Watch Time")
quiz_score = st.number_input("Quiz Score")
assignment_completion_rate = st.number_input("Assignment Completion Rate")
inactivity_days = st.number_input("Inactivity Days")

device_type = st.selectbox(
    "Device Type",
    ["mobile","laptop","tablet"]
)

internet_quality = st.selectbox(
    "Internet Quality",
    ["average","good","poor"]
)

study_environment = st.selectbox(
    "Study Environment",
    ["moderate","noisy","quiet"]
)

screen_time = st.number_input("Screen Time")

if st.button("Predict Focus Level"):

    device_map = {
        "laptop":0,
        "mobile":1,
        "tablet":2
    }

    internet_map = {
        "average":0,
        "good":1,
        "poor":2
    }

    study_map = {
        "moderate":0,
        "noisy":1,
        "quiet":2
    }

    data = np.array([[
        login_frequency,
        session_duration,
        video_watch_time,
        quiz_score,
        assignment_completion_rate,
        inactivity_days,
        device_map[device_type],
        internet_map[internet_quality],
        study_map[study_environment],
        screen_time
    ]])

    data = scaler.transform(data)
    prediction = model.predict(data)

    focus_map = {
        0:"High",
        1:"Low",
        2:"Medium"
    }

    
    result = focus_map[prediction[0]]
    
    if result == "High":
        st.success("🔥 High Focus Level")
        
    elif result == "Medium":
        st.warning("⚡ Medium Focus Level")
    else:
        st.error("😴 Low Focus Level")