# src/pipeline/train_pipeline.py

import os
import sys
from pathlib import Path

# --- THIS IS THE FIX ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
# --- END OF FIX ---

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    """
    This class manages the entire training process.
    It calls each component in the correct order.
    """
    def __init__(self):
        logging.info("Training Pipeline initialized.")

    def run_pipeline(self):
        logging.info("--- Starting Training Pipeline ---")
        try:
            # Step 1: Data Ingestion
            logging.info("Running Data Ingestion...")
            ingestor = DataIngestion()
            train_data_path, test_data_path = ingestor.initiate_data_ingestion()

            # Step 2: Data Transformation
            logging.info("Running Data Transformation...")
            transformer = DataTransformation()
            X_train_resampled, y_train_resampled, X_test, y_test, le = \
                transformer.initiate_data_transformation(train_data_path, test_data_path)

            # Step 3: Model Trainer
            logging.info("Running Model Trainer...")
            trainer = ModelTrainer()
            trainer.initiate_model_training(
                X_train_resampled, y_train_resampled, X_test, y_test, le
            )
            
            logging.info("--- Training Pipeline Finished Successfully ---")

        except Exception as e:
            logging.error(f"An error occurred in the training pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    logging.info("Running Training Pipeline as a standalone script...")
    pipeline = TrainPipeline()
    pipeline.run_pipeline()