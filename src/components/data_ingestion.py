# src/components/data_ingestion.py

import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# --- THIS IS THE FIX ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
# --- END OF FIX ---

from src.logger import logging # <-- Import the logger
from src.exception import CustomException # <-- Import the custom exception

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('data', 'dataset.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("DataIngestion component initialized.") # <-- Use logger

    def initiate_data_ingestion(self):
        logging.info("--- Starting Data Ingestion ---") # <-- Use logger
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Raw dataset read from {self.ingestion_config.raw_data_path}")
            df = df.sample(n=20000, random_state=42) # <-- ADD THIS LINE
            logging.info("Sampled 20,000 rows from the dataset.")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info(f"Created/verified 'artifacts' directory.")

            logging.info("Splitting data into train and test sets...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Data ingestion completed. Train and Test CSVs saved.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys) # <-- Use custom exception

if __name__ == "__main__":
    logging.info("Running Data Ingestion component as a script...")
    ingestor = DataIngestion()
    ingestor.initiate_data_ingestion()
    logging.info("Data Ingestion script finished.")