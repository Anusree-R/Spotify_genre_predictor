# app.py

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# --- THIS IS THE FIX ---
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
# --- END OF FIX ---

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

# --- 1. SET UP THE PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Spotify Genre Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- 2. ADD YOUR HEADER AND QUOTE ---
# Use a logo with a transparent background for dark mode
logo_svg = """
<div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px;">
    <svg width="50" height="50" viewBox="0 0 167.5 167.5" xmlns="http://www.w3.org/2000/svg" style="fill: #1DB954;">
        <path d="M83.7 0C37.5 0 0 37.5 0 83.7c0 46.3 37.5 83.7 83.7 83.7 46.3 0 83.7-37.5 83.7-83.7C167.5 37.5 130 0 83.7 0zM122 120.8c-1.4 2.5-4.4 3.2-6.8 1.8-17.8-10.9-40-13.4-66.6-7.4-2.7.6-5.5-1.1-6.1-3.7-.6-2.7 1.1-5.5 3.7-6.1 29.1-6.6 53.7-3.8 73.6 8.5 2.5 1.4 3.2 4.4 1.8 6.8zM133.4 98c-1.8 3.1-5.5 4.1-8.6 2.3-21-12.2-52.2-15.8-77.5-8.7-3.4.9-6.8-1.4-7.8-4.8-.9-3.4 1.4-6.8 4.8-7.8 28.1-7.8 62.4-3.9 86.4 10 3.1 1.8 4.1 5.5 2.3 8.6zM134.3 74.4c-26.2-14.1-69-15.1-80.5-8.5-4.1 2.3-9.1.9-11.4-3.1-2.3-4.1-.9-9.1 3.1-11.4 14.8-8.3 61.8-7.3 91.9 8.7 3.8 2 5.1 6.8 3.1 10.6-2 3.8-6.8 5.1-10.6 3.1z"></path>
    </svg>
    <h1 style="color: #1DB954; margin: 0; font-size: 2.2em; font-family: 'Montserrat', sans-serif;">Spotify Genre Predictor</h1>
</div>
"""
st.markdown(logo_svg, unsafe_allow_html=True)
#st.title("Spotify Genre Predictor")
st.markdown("*Music is the soundtrack of your life.* - Dick Clark")

st.markdown("---") # Adds a horizontal line

# --- 3. CREATE THE FORM ---
with st.form("prediction_form"):
    
    # We removed the "Numeric Features" header
    # Instead, we'll use 3 columns for a cleaner layout
    col1, col2, col3 = st.columns(3)
    
    # --- Column 1 ---
    with col1:
        st.subheader("Audio Properties")
        danceability = st.slider(
            "Danceability", 0.0, 1.0, 0.5, 0.01,
            help="How suitable a track is for dancing based on tempo, rhythm stability, etc."
        )
        energy = st.slider(
            "Energy", 0.0, 1.0, 0.5, 0.01,
            help="A perceptual measure of intensity and activity (e.g., fast, loud, noisy)."
        )
        loudness = st.slider(
            "Loudness (dB)", -60.0, 0.0, -5.0, 0.1,
            help="The overall loudness of a track in decibels (dB)."
        )

    # --- Column 2 ---
    with col2:
        st.subheader("Vocal Properties")
        speechiness = st.slider(
            "Speechiness", 0.0, 1.0, 0.1, 0.01,
            help="Detects the presence of spoken words in a track. Higher values mean more speech."
        )
        acousticness = st.slider(
            "Acousticness", 0.0, 1.0, 0.1, 0.01,
            help="A confidence measure from 0.0 to 1.0 of whether the track is acoustic."
        )
        instrumentalness = st.slider(
            "Instrumentalness", 0.0, 1.0, 0.0, 0.01,
            help="Predicts whether a track contains no vocals. Values closer to 1.0 are instrumental."
        )

    # --- Column 3 ---
    with col3:
        st.subheader("Live & Mood Properties")
        liveness = st.slider(
            "Liveness", 0.0, 1.0, 0.1, 0.01,
            help="Detects the presence of an audience in the recording."
        )
        valence = st.slider(
            "Valence", 0.0, 1.0, 0.5, 0.01,
            help="A measure of musical positiveness (e.g., happy, cheerful, euphoric)."
        )
        tempo = st.slider(
            "Tempo (BPM)", 0.0, 250.0, 120.0, 0.1,
            help="The speed or pace of a given piece, in beats per minute (BPM)."
        )

    st.markdown("---") # Horizontal line
    
    # We removed the "Categorical Features" header
    st.subheader("Track Characteristics")
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    
    with cat_col1:
        key = st.selectbox(
            "Key (0-11)", options=list(range(12)), index=5,
            help="The key the track is in (e.g., 0=C, 1=C#, 2=D, etc.)."
        )
        
    with cat_col2:
        mode = st.selectbox(
            "Mode (0=Minor, 1=Major)", options=[0, 1], index=1,
            help="The modality (Major or Minor) of a track."
        )
        
    with cat_col3:
        time_signature = st.selectbox(
            "Time Signature", options=[1, 3, 4, 5], index=2,
            help="An estimated time signature (e.g., 3 = 3/4, 4 = 4/4)."
        )

    st.markdown("---")
    
    # --- The "Predict" button ---
    submitted = st.form_submit_button("Predict Genre")

# --- 4. HANDLE THE PREDICTION (and the new result design) ---
if submitted:
    try:
        logging.info("Prediction button clicked. Creating CustomData.")
        data = CustomData(
            danceability=danceability, energy=energy, loudness=loudness,
            speechiness=speechiness, acousticness=acousticness, instrumentalness=instrumentalness,
            liveness=liveness, valence=valence, tempo=tempo,
            key=key, mode=mode, time_signature=time_signature
        )
        
        features_df = data.get_data_as_dataframe()
        logging.info("Data converted to DataFrame.")

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(features_df)
        
        logging.info(f"Prediction successful. Result: {result}")
        
        # --- NEW: Designed Result Box ---
        st.subheader("Predicted Genre:")
        st.markdown(f"""
        <div style="
            border: 2px solid #1DB954; 
            border-radius: 10px; 
            padding: 20px; 
            text-align: center;
            background-color: #282828;
        ">
            <h1 style="color: #1DB954; margin: 0;">{result}</h1>
        </div>
        """, unsafe_allow_html=True)
        # --- END OF NEW DESIGN ---
        
    except Exception as e:
        logging.error(f"An error occurred during Streamlit prediction: {e}")
        st.error(f"An error occurred: {e}")