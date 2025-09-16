# app.py
# The Flask application to serve the thyroid prediction model.

import flask
from flask import request, render_template
import numpy as np
import pickle
import os

# Create the Flask application instance
app = flask.Flask(__name__, template_folder='templates')

# Define the paths to the model files
MODEL_PATH = 'best_thyroid_model.pkl'
SCALER_PATH = 'thyroid_scaler.pkl'
CLASS_NAMES_PATH = 'thyroid_class_names.pkl'

# Load the pre-trained models and scaler
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    class_names = pickle.load(open(CLASS_NAMES_PATH, 'rb'))
    print("✅ Model files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    model, scaler, class_names = None, None, None

@app.route('/')
def home():
    """Renders the main page with the input form."""
    if not all([model, scaler, class_names]):
        return render_template('error.html', message="Server configuration error: Required model files are missing."), 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the web form.
    Extracts features, scales them, and uses the model to make a prediction.
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

        # Let the AI model make its prediction first
        prediction_index = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        predicted_class_name = class_names[prediction_index]
        confidence = probabilities[prediction_index] * 100

        # --- START OF SMARTER SAFETY NET ---
        # This logic trusts the model first, but corrects obvious clinical errors.
        # We use standard clinical reference ranges for TSH (approx. 0.4 to 5.0).
        
        # Check for potential Hyperthyroid conflict
        if inputs['basal_tsh'] < 0.4 and predicted_class_name != 'Hyperthyroid':
            # If TSH is clinically low but model didn't say Hyperthyroid, correct it.
            predicted_class_name = 'Hyperthyroid'
            confidence = 99.8 # Use a specific confidence to show it's a correction
            print("✅ Correction: TSH is clinically low. Overriding model's prediction to Hyperthyroid.")

        # Check for potential Hypothyroid conflict
        elif inputs['basal_tsh'] > 5.0 and predicted_class_name != 'Hypothyroid':
            # If TSH is clinically high but model didn't say Hypothyroid, correct it.
            predicted_class_name = 'Hypothyroid'
            confidence = 99.8
            print("✅ Correction: TSH is clinically high. Overriding model's prediction to Hypothyroid.")
        # --- END OF SMARTER SAFETY NET ---

        # Prepare data for the result template
        probs_dict = {name: f'{p * 100:.2f}' for name, p in zip(class_names, probabilities)}
        input_summary_dict = {
            'T3 Resin Uptake Test (%)': inputs['T3_resin_uptake'],
            'Total Serum Thyroxin (TT4)': inputs['total_serum_thyroxin'],
            'Total Serum Triiodothyronine (T3)': inputs['total_serum_t3'],
            'Basal TSH': inputs['basal_tsh'],
            'Max TSH Difference': inputs['max_diff_tsh']
        }

        return render_template('result.html',
                               prediction=predicted_class_name,
                               confidence=f'{confidence:.2f}',
                               probabilities=probs_dict,
                               inputs=input_summary_dict)

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return render_template('error.html', message="An unexpected server error occurred."), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)