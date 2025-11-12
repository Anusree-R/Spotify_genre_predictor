# src/utils.py

import os
import pickle
import sys
from src.logger import logging  # <-- Import the logger
from src.exception import CustomException # <-- Import the custom exception

def save_object(file_path, obj):
    """
    Saves a Python object as a pickle file, with logging and exceptions.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object saved to {file_path}") # <-- Use logger

    except Exception as e:
        # Raise our custom exception
        raise CustomException(e, sys)