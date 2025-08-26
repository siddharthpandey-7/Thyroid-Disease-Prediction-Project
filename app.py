# app.py
# The Flask application to serve the thyroid prediction model.
# This code is designed to load the pre-trained model and scaler
# and use them to make predictions based on user input from a web form.

import flask
from flask import request, render_template, jsonify
import numpy as np
import pickle
import os

# Create the Flask application instance
app = flask.Flask(__name__, template_folder='templates')

# Define the paths to the model files
MODEL_PATH = 'best_thyroid_model.pkl'
SCALER_PATH = 'thyroid_scaler.pkl'
CLASS_NAMES_PATH = 'thyroid_class_names.pkl'

# Initialize model objects as None. They will be loaded when the app starts.
model = None
scaler = None
class_names = None

def load_models_and_scaler():
    """
    Loads the pre-trained model, scaler, and class names from pickle files.
    This function is called once when the application starts to improve performance.
    """
    global model, scaler, class_names
    try:
        print("Attempting to load model files...")
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(SCALER_PATH, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        with open(CLASS_NAMES_PATH, 'rb') as names_file:
            class_names = pickle.load(names_file)
        print("✅ Model files loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: Model file not found. Please ensure all files are in the correct directory. {e}")
        model, scaler, class_names = None, None, None
    except Exception as e:
        print(f"❌ Error loading model files: {e}")
        model, scaler, class_names = None, None, None

# Load the models when the application starts.
# This replaces the now-deprecated @app.before_first_request decorator.
load_models_and_scaler()

@app.route('/')
def home():
    """Renders the main page with the input form."""
    # Check if all necessary files were loaded successfully
    if not all([model, scaler, class_names]):
        return render_template('error.html', message="Server configuration error: Required model files are missing."), 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the web form.
    It extracts the features, scales them, and uses the model to make a prediction.
    """
    if not all([model, scaler, class_names]):
        return render_template('error.html', message="Server configuration error: Model is not available for predictions."), 500

    try:
        # Extract and validate numerical inputs from the form
        inputs = {
            'T3_resin_uptake': float(request.form['t3_resin_uptake']),
            'total_serum_thyroxin': float(request.form['total_serum_thyroxin']),
            'total_serum_t3': float(request.form['total_serum_t3']),
            'basal_tsh': float(request.form['basal_tsh']),
            'max_diff_tsh': float(request.form['max_diff_tsh'])
        }

        # Reshape the input data for the model
        features_array = np.array(list(inputs.values())).reshape(1, -1)
        
        # Scale the features using the pre-trained scaler
        scaled_features = scaler.transform(features_array)

        # Make the prediction
        prediction_index = model.predict(scaled_features)[0]
        
        # Get the class probabilities
        probabilities = model.predict_proba(scaled_features)[0]

        # Map the prediction index and probabilities to human-readable names
        predicted_class_name = class_names[prediction_index]
        confidence = probabilities[prediction_index] * 100

        # --- START OF RULE-BASED OVERRIDE (THE FIX) ---
        # A very low TSH value is a strong indicator of Hyperthyroidism
        if inputs['basal_tsh'] < 0.2:
            override_prediction = 'Hyperthyroid'
            override_confidence = 100.00
            # Find the index for Hyperthyroid and update probabilities
            hyper_index = list(class_names).index('Hyperthyroid')
            probabilities = np.zeros_like(probabilities)
            probabilities[hyper_index] = 1.0
            print(f"✅ Override: TSH is very low ({inputs['basal_tsh']}), correcting to Hyperthyroid.")
            predicted_class_name = override_prediction
            confidence = override_confidence

        # A very high TSH value is a strong indicator of Hypothyroidism
        elif inputs['basal_tsh'] > 10.0:
            override_prediction = 'Hypothyroid'
            override_confidence = 100.00
            # Find the index for Hypothyroid and update probabilities
            hypo_index = list(class_names).index('Hypothyroid')
            probabilities = np.zeros_like(probabilities)
            probabilities[hypo_index] = 1.0
            print(f"✅ Override: TSH is very high ({inputs['basal_tsh']}), correcting to Hypothyroid.")
            predicted_class_name = override_prediction
            confidence = override_confidence
        # --- END OF RULE-BASED OVERRIDE ---

        # Prepare data for the result template
        probs_dict = {class_names[i]: f'{p * 100:.2f}' for i, p in enumerate(probabilities)}
        input_summary_dict = {
            'T3 Resin Uptake Test (%)': inputs['T3_resin_uptake'],
            'Total Serum Thyroxin (TT4)': inputs['total_serum_thyroxin'],
            'Total Serum Triiodothyronine (T3)': inputs['total_serum_t3'],
            'Basal TSH': inputs['basal_tsh'],
            'Max TSH Difference (Post Stimulation)': inputs['max_diff_tsh']
        }

        # Render the result template with all the processed data
        return render_template('result.html',
                               prediction=predicted_class_name,
                               confidence=f'{confidence:.2f}',
                               probabilities=probs_dict,
                               inputs=input_summary_dict)

    except (ValueError, KeyError) as e:
        # Handle cases where the input is not a valid number or a key is missing
        print(f"❌ Input validation error: {e}")
        return render_template('error.html', message="Invalid input: Please enter valid numbers for all fields."), 400
    except Exception as e:
        # Catch any other unexpected errors
        print(f"❌ An unexpected error occurred: {e}")
        return render_template('error.html', message="An unexpected server error occurred. Please try again later."), 500

# Custom error handlers for common HTTP errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="The page you are looking for does not exist."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', message="Internal Server Error: Something went wrong on the server."), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
