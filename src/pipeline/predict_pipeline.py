# src/pipeline/predict_pipeline.py

import os
import sys
from pathlib import Path
import pandas as pd
import pickle

# --- THIS IS THE FIX ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
# --- END OF FIX ---

from src.logger import logging
from src.exception import CustomException

class CustomData:
    """
    This class takes the data from the Streamlit sliders.
    """
    def __init__(self,
                 danceability: float, energy: float, loudness: float,
                 speechiness: float, acousticness: float, instrumentalness: float,
                 liveness: float, valence: float, tempo: float,
                 key: int, mode: int, time_signature: int):
        
        self.danceability = danceability
        self.energy = energy
        self.loudness = loudness
        self.speechiness = speechiness
        self.acousticness = acousticness
        self.instrumentalness = instrumentalness
        self.liveness = liveness
        self.valence = valence
        self.tempo = tempo
        self.key = key
        self.mode = mode
        self.time_signature = time_signature

    def get_data_as_dataframe(self):
        """
        Converts the slider data into a single-row DataFrame
        """
        try:
            custom_data_input_dict = {
                "danceability": [self.danceability], "energy": [self.energy],
                "loudness": [self.loudness], "speechiness": [self.speechiness],
                "acousticness": [self.acousticness], "instrumentalness": [self.instrumentalness],
                "liveness": [self.liveness], "valence": [self.valence],
                "tempo": [self.tempo], "key": [self.key],
                "mode": [self.mode], "time_signature": [self.time_signature],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("CustomData (from sliders) converted to DataFrame.")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)

class PredictPipeline:
    """
    This is the "brain" for the slider app.
    It predicts for ONE song at a time.
    """
    def __init__(self):
        logging.info("PredictPipeline initialized.")
        self.model_path = os.path.join("artifacts", "spotify_genre_model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")

    def predict(self, features_df):
        """
        Takes one row of data and returns ONE genre and ONE confidence.
        """
        try:
            logging.info("Starting single prediction...")
            
            model = pickle.load(open(self.model_path, "rb"))
            preprocessor = pickle.load(open(self.preprocessor_path, "rb"))
            label_encoder = pickle.load(open(self.label_encoder_path, "rb"))
            logging.info("All artifacts loaded.")

            processed_data = preprocessor.transform(features_df)
            logging.info("Data transformed.")

            probabilities = model.predict_proba(processed_data)

            confidence = probabilities.max() * 100 # This is a single number
            prediction_encoded = [probabilities.argmax()]
            predicted_genre = label_encoder.inverse_transform(prediction_encoded)
            logging.info(f"Decoded prediction: {predicted_genre[0]} with {confidence:.2f}% confidence.")

            # Return single values, NOT lists. This fixes the error.
            return predicted_genre[0], confidence 

        except Exception as e:
            raise CustomException(e, sys)