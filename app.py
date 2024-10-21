from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = 'aqi_model.pkl'  # Ensure this path is correct in the cloud environment
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert JavaScript variable names back to original names for the model
    data['PM2.5'] = data.pop('PM25')  # Convert PM25 to PM2.5

    # Extract features from the JSON payload
    try:
        CO = data['CO']
        NO2 = data['NO2']
        O3 = data['O3']
        SO2 = data['SO2']
        PM25 = data['PM2.5']
        PM10 = data['PM10']
        NH3 = data['NH3']

        # Create a numpy array with the correct order of features
        features = np.array([[CO, NO2, O3, SO2, PM25, PM10, NH3]])

        # Make prediction using the loaded model
        prediction = model.predict(features)[0]

        # Return the predicted AQI
        return jsonify({'AQI': prediction})

    except KeyError as e:
        return jsonify({'error': f'Missing data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
