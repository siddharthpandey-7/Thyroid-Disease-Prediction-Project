🧪 Thyroid Disease Prediction System

An end-to-end machine learning–based web application that predicts thyroid conditions (Normal, Hyperthyroid, Hypothyroid) using clinical thyroid function test parameters.
The system combines classical machine learning, medical domain rules, and a Flask-based web interface to provide reliable and interpretable predictions.

🎯 Project Overview

This project addresses the problem of thyroid disease classification using structured medical data.
It takes five thyroid-related blood test values as input and predicts the thyroid condition with a confidence score and probability breakdown.

The system uses a trained XGBoost classifier, applies consistent preprocessing using a saved scaler, and includes a clinical safety override based on TSH thresholds to improve medical reliability.

✅ Key Highlights

End-to-end ML pipeline (data → model → deployment)

Multiple model comparison before final selection

Class imbalance handled using SMOTE

Robust preprocessing with StandardScaler

Flask-based web deployment

Confidence scores and probability visualization

Rule-based clinical override for extreme TSH values

Graceful error handling with a custom error page

🏥 Medical Background
Conditions Predicted

Normal – Healthy thyroid function

Hyperthyroid – Overactive thyroid

Hypothyroid – Underactive thyroid

Input Parameters

The model uses the following five clinical features:

T3 Resin Uptake Test (%)

Total Serum Thyroxin (TT4)

Total Serum Triiodothyronine (T3)

Basal TSH

Max TSH Difference (post stimulation)

🧠 Machine Learning Pipeline
Dataset

Total samples: 215

Features: 5 clinical parameters

Target classes: 3 (Normal, Hyperthyroid, Hypothyroid)

Data Preprocessing

Missing values removed

All features converted to numeric format

Feature scaling using StandardScaler

Class imbalance handled using SMOTE

Train–test split: 80% training / 20% testing with stratification

Models Trained and Evaluated

The following models were trained and compared:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

XGBoost

Evaluation metrics:

Accuracy

Weighted F1-score

Classification report

Confusion matrix

Model Selection

Although Logistic Regression and SVM achieved perfect accuracy on the test set, XGBoost was selected because:

It captures non-linear relationships better

It is more robust for structured medical data

It is less likely to overfit small datasets

Final Model Performance (Test Set)

XGBoost Classifier

Accuracy: 97.67%

Weighted F1-score: 0.9760

Classification Report:

                 precision    recall  f1-score   support
Normal              0.97      1.00      0.98        30
Hyperthyroid        1.00      0.86      0.92         7
Hypothyroid         1.00      1.00      1.00         6

accuracy                                0.98        43
macro avg           0.99      0.95      0.97        43
weighted avg        0.98      0.98      0.98        43

🩺 Clinical Safety Override

To improve medical reliability, a rule-based override is applied after ML prediction:

Basal TSH < 0.4 → Hyperthyroid

Basal TSH > 5.0 → Hypothyroid

When triggered:

ML prediction is overridden

Confidence is set to 99.8%

Probability distribution is adjusted accordingly

This ensures that clinical domain rules take precedence over ML in extreme cases.

🌐 Web Application Architecture
Backend

Framework: Flask

Loads trained model, scaler, and class labels

Handles preprocessing, prediction, and rule-based logic

Uses POST requests for secure data transfer

Includes try–except–based error handling

Frontend

HTML templates rendered using Flask (Jinja)

Styled with Tailwind CSS

User-friendly input form

Color-coded prediction results

Probability bars and confidence display

Downloadable prediction report

📁 Project Structure

thyroid-disease-prediction/
│
├── app.py                         # Flask backend application
├── Thyroid.ipynb                  # Model training & evaluation notebook
├── thyroid_dataset.csv            # Dataset
│
├── templates/
│   ├── index.html                 # Input form
│   ├── result.html                # Prediction results
│   └── error.html                 # Error handling page
│
├── best_thyroid_model.pkl         # Trained XGBoost model
├── thyroid_scaler.pkl             # StandardScaler object
├── thyroid_class_names.pkl        # Class label mapping
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore


⚙️ How to Run the Project
Prerequisites

Python 3.8+

pip

Installation & Execution

# Clone repository
git clone https://github.com/yourusername/thyroid-disease-prediction.git
cd thyroid-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py

Open browser:

http://localhost:5000

🔧 Technologies Used

Python

Scikit-learn

XGBoost

Imbalanced-learn (SMOTE)

Flask

NumPy & Pandas

Matplotlib & Seaborn

HTML, Tailwind CSS, JavaScript

Pickle (model persistence)

🎯 Design Decisions (Important)

Chose XGBoost for robustness over perfect accuracy

Used SMOTE to handle imbalanced medical data

Saved scaler separately to ensure preprocessing consistency

Added clinical rule override for safety-critical predictions

Focused on deployment, not just notebook-based ML

⚠️ Medical Disclaimer

This project is for educational and research purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Always consult qualified healthcare professionals for medical decisions.

📞 Support

For questions, issues, or suggestions related to this project:

Issues: Use the GitHub Issues section of this repository

Discussions: Use GitHub Discussions for general questions and ideas

Email: your.email@example.com

Contributions, feedback, and improvements are always welcome.

Built with ❤️ for better healthcare through AI
