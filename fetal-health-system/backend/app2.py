from flask import Flask, request, render_template
import pickle
import numpy as np

# Update the template folder path to point to the correct location
app = Flask(__name__, template_folder='D:/Fetal Health 2/fetal-health-system/templates')

# Load the model and scaler
with open('D:/Fetal Health 2/fetal-health-system/backend/model/fetal_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('D:/Fetal Health 2/fetal-health-system/backend/model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Helper function to check if prediction is pathogenic
def is_pathogenic(prediction):
    return prediction == 3  # Assuming '3' indicates pathogenic fetal health

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    accuracy = "Not yet predicted"
    if request.method == 'POST':
        try:
            # Collect input features from the form
            features = [float(request.form[f'feature{i}']) for i in range(1, 22)]
            
            # Convert to NumPy array and reshape for the model
            features_array = np.array(features).reshape(1, -1)
            
            # Scale the input features
            scaled_features = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            
            # Check if the result is pathogenic
            if is_pathogenic(prediction):
                response_message = "Warning: Pathogenic reading detected! Immediate action required."
            else:
                if prediction == 1:
                    response_message = "Fetal health is normal."
                else:
                    response_message = "Fetal health is suspicious, further monitoring is recommended."
            
            accuracy = response_message
        except Exception as e:
            accuracy = f"Error: {str(e)}"
    
    return render_template('index2.html', prediction=prediction, accuracy=accuracy)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
