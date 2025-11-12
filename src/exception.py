# src/exception.py

import sys
import os
from src.logger import logging  # <-- It imports our new logger!

def get_error_details(error, error_detail: sys):
    """
    This function creates a detailed error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
    
    return error_message

class CustomException(Exception):
    """
    This is our custom exception class.
    When we 'raise' this exception, it will:
    1. Create a detailed error message.
    2. Write that message to our log file.
    """
    def __init__(self, error_message, error_detail: sys):
        # We inherit from the base Exception class
        super().__init__(error_message)
        
        # Get the detailed error message
        self.detailed_error_message = get_error_details(error=error_message, error_detail=error_detail)
        
        # Log the detailed error message
        logging.error(self.detailed_error_message)

    def __str__(self):
        return self.detailed_error_message

# This is just for testing
if __name__ == "__main__":
    try:
        a = 1/0 # Create a deliberate error
    except Exception as e:
        # Log the error
        logging.info("A divide by zero error occurred.")
        # Raise our custom exception
        raise CustomException(e, sys)