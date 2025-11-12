# src/components/model_trainer.py

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- THIS IS THE FIX ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
# --- END OF FIX ---

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "spotify_genre_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        logging.info("ModelTrainer component initialized.")

    # --- THIS FUNCTION IS NOW MODIFIED ---
    def initiate_model_training(self, X_train, y_train, X_test, y_test, le): #<-- Vars renamed
        logging.info("--- Starting Model Training ---")
        try:
            logging.info("Data for training has been received.")

            # --- 2. Initialize the model (WITH NEW SETTING) ---
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',  # <-- THIS IS THE FIX FOR THE 909MB FILE
                n_jobs=-1
            )
            logging.info("RandomForestClassifier model initialized with class_weight='balanced'.")

            # --- 3. Train the model ---
            logging.info("Training the model...")
            model.fit(X_train, y_train) #<-- Use original processed data
            logging.info("Model training complete.")

            # --- 4. Evaluate the model ---
            logging.info("Evaluating model on the test set...")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le.classes_)

            logging.info(f"Model Accuracy on Test Set: {accuracy:.2%}")
            logging.info(f"Classification Report:\n{report}")
            
            # --- 5. Save the final, trained model ---
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info("--- Model Training Complete. Trained model saved to 'artifacts'. ---")

        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise CustomException(e, sys)

# --- THIS BLOCK IS ALSO UPDATED ---
if __name__ == "__main__":
    logging.info("Running Model Trainer as a standalone script...")
    
    ingestor = DataIngestion()
    train_path, test_path = ingestor.initiate_data_ingestion()
    
    transformer = DataTransformation()
    # Get the new, non-resampled data
    X_train, y_train, X_test, y_test, le = \
        transformer.initiate_data_transformation(train_path, test_path)
    
    trainer = ModelTrainer()
    trainer.initiate_model_training(X_train, y_train, X_test, y_test, le) # Pass new data