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
    This class defines the exact data structure that the
    web form (app.py) will send to the pipeline.
    """
    def __init__(self,
                 danceability: float,
                 energy: float,
                 loudness: float,
                 speechiness: float,
                 acousticness: float,
                 instrumentalness: float,
                 liveness: float,
                 valence: float,
                 tempo: float,
                 key: int,
                 mode: int,
                 time_signature: int):
        
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
        Converts the user's input data into a single-row DataFrame
        that our model's preprocessor can understand.
        """
        try:
            custom_data_input_dict = {
                "danceability": [self.danceability],
                "energy": [self.energy],
                "loudness": [self.loudness],
                "speechiness": [self.speechiness],
                "acousticness": [self.acousticness],
                "instrumentalness": [self.instrumentalness],
                "liveness": [self.liveness],
                "valence": [self.valence],
                "tempo": [self.tempo],
                "key": [self.key],
                "mode": [self.mode],
                "time_signature": [self.time_signature],
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("CustomData converted to DataFrame successfully.")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)

class PredictPipeline:
    """
    This class loads all the saved artifacts (model, preprocessor, etc.)
    and uses them to make a prediction on new data.
    """
    def __init__(self):
        logging.info("PredictPipeline initialized.")
        # Define the paths to all our saved artifacts
        self.model_path = os.path.join("artifacts", "spotify_genre_model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")

    def predict(self, features_df):
        """
        Takes a DataFrame of new data, applies all transformations,
        and returns a predicted genre name.
        """
        try:
            logging.info("Starting prediction...")
            
            # Load the saved objects
            model = pickle.load(open(self.model_path, "rb"))
            preprocessor = pickle.load(open(self.preprocessor_path, "rb"))
            label_encoder = pickle.load(open(self.label_encoder_path, "rb"))
            logging.info("All artifacts loaded successfully.")

            # Apply the preprocessor (scaling, one-hot encoding)
            processed_data = preprocessor.transform(features_df)
            logging.info("Data transformed using saved preprocessor.")

            # Make the prediction
            prediction_encoded = model.predict(processed_data)
            logging.info(f"Encoded prediction: {prediction_encoded}")

            # Decode the prediction (e.g., turn '5' into 'Pop')
            predicted_genre = label_encoder.inverse_transform(prediction_encoded)
            logging.info(f"Decoded prediction: {predicted_genre[0]}")

            return predicted_genre[0]

        except Exception as e:
            raise CustomException(e, sys)

# This block is for testing
if __name__ == '__main__':
    logging.info("Running PredictPipeline as a standalone script...")
    
    # Create a sample song (example: Pop-like song)
    sample_data = CustomData(
        danceability=0.7, energy=0.8, loudness=-5.0, speechiness=0.1,
        acousticness=0.2, instrumentalness=0.0, liveness=0.15,
        valence=0.6, tempo=120.0, key=5, mode=1, time_signature=4
    )
    
    # Convert to DataFrame
    sample_df = sample_data.get_data_as_dataframe()
    
    # Get a prediction
    pipeline = PredictPipeline()
    result = pipeline.predict(sample_df)
    
    print(f"Prediction test successful. Predicted genre: {result}")