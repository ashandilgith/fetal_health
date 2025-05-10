from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)



# Load the model and scaler
with open('D:/Fetal Health 2/fetal-health-system/backend/model/fetal_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('D:/Fetal Health 2/fetal-health-system/backend/model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Helper function to check if prediction is pathogenic
def is_pathogenic(prediction):
    return prediction == 3  # Assuming '3' indicates pathogenic fetal health

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    
    # Scale the input features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    
    # Check for pathogenic readings
    if is_pathogenic(prediction):
        warning_message = "Warning: Pathogenic reading detected! Immediate action required."
        print(warning_message)  # Print to console
        return jsonify({'prediction': int(prediction), 'alert': warning_message})
    
    # Response for non-pathogenic readings
    if prediction == 1:
        response_message = "Fetal health is normal."
    else:  # prediction == 2
        response_message = "Fetal health is suspicious, further monitoring is recommended."
    
    return jsonify({'prediction': int(prediction), 'alert': response_message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
