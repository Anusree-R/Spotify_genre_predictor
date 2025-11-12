# src/components/data_transformation.py

import os
import sys
from pathlib import Path

# --- THIS IS THE FIX ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
# --- END OF FIX ---

import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE  <-- WE DON'T NEED THIS ANYMORE

# --- Import our new tools ---
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.components.data_ingestion import DataIngestionConfig, DataIngestion
# --- End of new imports ---

# (The consolidate_genre_improved function is unchanged)
def consolidate_genre_improved(genre):
    genre = str(genre).lower()
    if 'rock' in genre: return 'Rock'
    if 'pop' in genre or 'pop-film' in genre: return 'Pop'
    if 'hip-hop' in genre or 'rap' in genre: return 'Hip-Hop'
    if 'r-n-b' in genre or 'soul' in genre or 'funk' in genre: return 'R&B/Soul/Funk'
    if 'techno' in genre or 'trance' in genre or 'house' in genre or 'edm' in genre or \
       'electro' in genre or 'electronic' in genre or 'dubstep' in genre or \
       'drum-and-bass' in genre: return 'Electronic'
    if 'classical' in genre or 'opera' in genre: return 'Classical'
    if 'jazz' in genre or 'bossanova' in genre: return 'Jazz'
    if 'acoustic' in genre or 'folk' in genre or 'bluegrass' in genre: return 'Folk/Acoustic'
    if 'metal' in genre: return 'Metal'
    if 'latin' in genre or 'latino' in genre or 'salsa' in genre or 'samba' in genre or \
       'reggaeton' in genre: return 'Latin'
    if 'reggae' in genre or 'ska' in genre: return 'Reggae/Ska'
    if 'indie' in genre: return 'Indie'
    if 'world-music' in genre or 'indian' in genre or 'malay' in genre or 'mandopop' in genre or \
       'j-pop' in genre or 'k-pop' in genre or 'turkish' in genre: return 'World Music'
    return 'Other'

# (The DataTransformationConfig class is unchanged)
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    label_encoder_obj_file_path: str = os.path.join('artifacts', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self.data_ingestion_config = DataIngestionConfig()
        logging.info("DataTransformation component initialized.")

    # (The get_data_transformer_object function is unchanged)
    def get_data_transformer_object(self):
        logging.info("Creating data transformer object...")
        try:
            numeric_features = [
                'danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
            ]
            categorical_features = ['key', 'mode', 'time_signature']
            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))])
            logging.info("Numeric and categorical pipelines created.")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ],
                remainder='drop' 
            )
            logging.info("ColumnTransformer preprocessor created.")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    # --- THIS FUNCTION IS NOW MODIFIED ---
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("--- Starting Data Transformation ---")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("train.csv and test.csv read successfully.")

            train_df['consolidated_genre'] = train_df['track_genre'].apply(consolidate_genre_improved)
            test_df['consolidated_genre'] = test_df['track_genre'].apply(consolidate_genre_improved)
            train_df = train_df[train_df['consolidated_genre'] != 'Other']
            test_df = test_df[test_df['consolidated_genre'] != 'Other']
            logging.info("Genre consolidated and 'Other' category removed.")

            target_column_name = 'consolidated_genre'
            X_train = train_df.drop(columns=[target_column_name, 'track_genre'], axis=1)
            y_train = train_df[target_column_name]
            X_test = test_df.drop(columns=[target_column_name, 'track_genre'], axis=1)
            y_test = test_df[target_column_name]
            logging.info("X and y separated for train and test sets.")

            preprocessor = self.get_data_transformer_object()

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            logging.info("Target variable (y) encoded.")

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            logging.info("Preprocessor applied to X_train and X_test.")
            
            # --- SMOTE IS REMOVED ---
            # logging.info("SMOTE applied to training data.")
            # --- END OF REMOVAL ---

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            save_object(
                file_path=self.transformation_config.label_encoder_obj_file_path,
                obj=le
            )
            logging.info("Saved preprocessor and label encoder objects.")
            logging.info("--- Data Transformation Complete. ---")
            
            # We now return the ORIGINAL processed data, not the resampled data
            return (
                X_train_processed, #<-- CHANGED
                y_train_encoded,   #<-- CHANGED
                X_test_processed,
                y_test_encoded,
                le
            )

        except Exception as e:
            logging.error(f"An error occurred during data transformation: {e}")
            raise CustomException(e, sys)

# (The if __name__ block is unchanged)
if __name__ == "__main__":
    logging.info("Running Data Transformation as a standalone script...")
    ingestor = DataIngestion()
    train_path, test_path = ingestor.initiate_data_ingestion()
    transformer = DataTransformation()
    transformer.initiate_data_transformation(train_path, test_path)