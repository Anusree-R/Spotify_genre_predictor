# src/logger.py

import logging
import os
from datetime import datetime

# Define the log file name based on the current timestamp
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path to the 'logs' folder, which will be in the root directory
LOGS_DIR = os.path.join(os.getcwd(), "logs")

# Create the 'logs' folder if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE_NAME)

# --- Configure the logging ---
# This is the core of the logger.
# It sets the format for the log messages and the file to write to.

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

# This is just for testing the logger
if __name__ == "__main__":
    logging.info("Logging has started.")