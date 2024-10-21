# Import necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('aqi_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json

        # Extract features and preprocess
        features = np.array([
            data['CO'],
            data['NO2'] / 1000,  # Convert NO2 to mg/m³
            data['O3'] / 1000,   # Convert O3 to mg/m³
            data['SO2'] / 1000,  # Convert SO2 to mg/m³
            data['PM2.5'] / 1000,  # Convert PM2.5 to mg/m³
            data['PM10'] / 1000,  # Convert PM10 to mg/m³
            data['NH3'] / 1000    # Convert NH3 to mg/m³
        ]).reshape(1, -1)  # Reshape to 2D array for prediction

        # Predict using the model
        prediction = model.predict(features)

        # Return the prediction as a JSON response
        return jsonify({'AQI': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
