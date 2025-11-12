# app.py

import os
import sys
from pathlib import Path

# --- THIS IS THE FIX ---
# This code adds your main project folder ('Spotify_genre_predictor')
# to Python's path so it can find the 'src' module
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
# --- END OF FIX ---

from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException

# Initialize the Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home_page():
    return render_template('home.html')

# Route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        if request.method == 'POST':
            logging.info("Prediction request received.")

            # 1. Get all the data from the HTML form
            data = CustomData(
                danceability=float(request.form.get('danceability')),
                energy=float(request.form.get('energy')),
                loudness=float(request.form.get('loudness')),
                speechiness=float(request.form.get('speechiness')),
                acousticness=float(request.form.get('acousticness')),
                instrumentalness=float(request.form.get('instrumentalness')),
                liveness=float(request.form.get('liveness')),
                valence=float(request.form.get('valence')),
                tempo=float(request.form.get('tempo')),
                key=int(request.form.get('key')),
                mode=int(request.form.get('mode')),
                time_signature=int(request.form.get('time_signature'))
            )
            
            # 2. Convert the data into a DataFrame
            features_df = data.get_data_as_dataframe()

            # 3. Call the prediction pipeline
            predict_pipeline = PredictPipeline()
            result = predict_pipeline.predict(features_df)
            
            logging.info(f"Prediction complete. Result: {result}")

            # 4. Send the result back to the home.html page
            return render_template('home.html', prediction_result=f"Predicted Genre: {result}")

    except Exception as e:
        logging.error(f"An error occurred in the /predict route: {e}")
        raise CustomException(e, sys)

# This block allows you to run the app from the terminal
if __name__ == "__main__":
    logging.info("Starting Flask application...")
    app.run(debug=True, port=8088) # Use this for testing
    #app.run(host="0.0.0.0", port=8080) # Use this for deployment