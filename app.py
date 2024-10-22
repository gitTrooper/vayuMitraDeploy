from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('aqi_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Create a list of input features
    input_features = [data['CO'], data['NO2']/1000, data['O3']/1000, data['SO2']/1000, data['PM2.5']/1000, data['PM10']/1000, data['NH3']/1000]

    # Make predictions
    prediction = model.predict([input_features])

    # Return the predicted AQI value
    return jsonify({'AQI': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
      
