from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    model_metadata = pickle.load(open('model_metadata.pkl', 'rb'))
    feature_names = model_metadata['feature_names']
    scaler = model_metadata['scaler']
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please run train_model.py first.")
    model = None
    feature_names = []
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    try:
        # Get data from POST request
        data = request.get_json(force=True)
        
        # Create a DataFrame with the correct feature order
        input_data = pd.DataFrame(columns=feature_names)
        input_data.loc[0] = 0  # Initialize with zeros
        
        # Fill in the values we have
        for feature in data:
            if feature in feature_names:
                input_data[feature] = data[feature]
        
        # For categorical features, set the appropriate dummy variable
        if 'Gender' in data:
            input_data['Gender_Male'] = 1 if data['Gender'] == 'Male' else 0
        
        if 'Married' in data:
            input_data['Married_Yes'] = 1 if data['Married'] == 'Yes' else 0
        
        if 'Education' in data:
            input_data['Education_Graduate'] = 1 if data['Education'] == 'Graduate' else 0
        
        if 'Self_Employed' in data:
            input_data['Self_Employed_Yes'] = 1 if data['Self_Employed'] == 'Yes' else 0
        
        if 'Property_Area' in data:
            if data['Property_Area'] == 'Rural':
                input_data['Property_Area_Rural'] = 1
            elif data['Property_Area'] == 'Semiurban':
                input_data['Property_Area_Semiurban'] = 1
            elif data['Property_Area'] == 'Urban':
                input_data['Property_Area_Urban'] = 1
        
        if 'Dependents' in data:
            if data['Dependents'] == '0':
                input_data['Dependents_0'] = 1
            elif data['Dependents'] == '1':
                input_data['Dependents_1'] = 1
            elif data['Dependents'] == '2':
                input_data['Dependents_2'] = 1
            elif data['Dependents'] == '3+':
                input_data['Dependents_3+'] = 1
        
        # Scale the features
        scaled_features = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Return prediction
        return jsonify({'loan_approval': bool(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
