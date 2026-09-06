# 🧪 Thyroid Disease Prediction System

An end-to-end machine learning web application that classifies thyroid conditions as **Normal, Hyperthyroid, or Hypothyroid** using five thyroid-related clinical parameters.

The project covers the complete machine learning workflow, including data preprocessing, class balancing with SMOTE, model comparison, model evaluation, model persistence, and deployment through a Flask web application.

> **Note:** This project is intended for educational and research purposes only and is not a medical diagnostic tool.

---

## 🎯 Project Overview

The **Thyroid Disease Prediction System** uses structured thyroid-related clinical data to classify an input into one of three categories:

- **Normal**
- **Hyperthyroid**
- **Hypothyroid**

Multiple machine learning classification algorithms are trained and evaluated, with **XGBoost** selected as the final model for deployment.

The trained model, feature scaler, and class-label mapping are saved using Python's `pickle` module and loaded by the Flask application to generate predictions through a web interface.

The application also includes a simple **rule-based TSH override** for very low or very high Basal TSH values.

---

## ✨ Key Features

- End-to-end machine learning pipeline
- Data cleaning and preprocessing
- Stratified train-test split
- Feature scaling using `StandardScaler`
- Class balancing using `SMOTE`
- Comparison of multiple machine learning models
- XGBoost-based final prediction model
- Prediction probability display
- Prediction confidence display
- Rule-based TSH prediction override
- Flask-based web application
- Responsive web interface
- Custom error handling page
- Downloadable prediction report
- Saved model, scaler, and class-label mapping

---

## 🏥 Prediction Classes

The system predicts one of the following three classes:

| Class | Description |
|---|---|
| Normal | Normal thyroid classification |
| Hyperthyroid | Hyperthyroid classification |
| Hypothyroid | Hypothyroid classification |

---

## 🧪 Input Features

The application accepts five clinical parameters:

1. **T3 Resin Uptake Test**
2. **Total Serum Thyroxin (TT4)**
3. **Total Serum Triiodothyronine (T3)**
4. **Basal TSH**
5. **Max TSH Difference**

These five features are used as inputs to the trained classification model.

---

## 📊 Dataset

The dataset contains:

- **215 samples**
- **5 input features**
- **3 target classes**

The original target labels are:

```text
1 → Normal
2 → Hyperthyroid
3 → Hypothyroid
```

For machine learning, the target labels are encoded as:

```text
0 → Normal
1 → Hyperthyroid
2 → Hypothyroid
```

---

## 🔄 Machine Learning Pipeline

The project follows the following workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature & Target Separation
     ↓
Target Encoding
     ↓
Stratified Train-Test Split
     ↓
Feature Scaling
     ↓
SMOTE on Training Data
     ↓
Model Training
     ↓
Model Comparison
     ↓
XGBoost Selection
     ↓
Model Persistence
     ↓
Flask Deployment
```

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

- Missing values represented by `?` are converted to `NaN`
- Rows containing missing values are removed
- Feature values are converted to numeric format
- Target labels are converted to integer values
- Target classes are encoded into numerical labels
- Data is split into training and testing sets using an 80/20 stratified split
- Features are standardized using `StandardScaler`
- SMOTE is applied to the training data only to address class imbalance

The test set is kept separate from SMOTE so that the evaluation data is not artificially modified.

---

## 🤖 Machine Learning Models

The following classification algorithms are trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

### Evaluation Metrics

The models are evaluated using:

- Accuracy
- Weighted F1-score
- Classification report
- Confusion matrix

---

## 🏆 Final Model Selection

XGBoost was selected as the final model for deployment.

The model was selected manually rather than automatically choosing the model with the highest test-set accuracy. The project uses XGBoost as the final model because it provides a nonlinear tree-based approach suitable for structured/tabular data.

The selection was made with consideration of model characteristics rather than relying solely on the accuracy obtained from a single test split.

---

## 📈 XGBoost Test Performance

The selected XGBoost model achieved the following results on the held-out test set:

- **Accuracy:** 97.67%
- **Weighted F1-score:** 0.9760

### Classification Report

```text
                 precision    recall  f1-score   support

Normal              0.97      1.00      0.98        30
Hyperthyroid        1.00      0.86      0.92         7
Hypothyroid         1.00      1.00      1.00         6

accuracy                                0.98        43
macro avg           0.99      0.95      0.97        43
weighted avg        0.98      0.98      0.98        43
```

> **Note:** The reported performance is based on a single train-test split of a relatively small dataset. It should not be interpreted as evidence of clinical performance or real-world diagnostic accuracy.

---

## 🩺 Rule-Based TSH Override

In addition to the machine learning prediction, the Flask application contains a simple rule-based override based on the Basal TSH value.

### Rules

```text
Basal TSH < 0.4  → Hyperthyroid
Basal TSH > 5.0  → Hypothyroid
```

If one of these conditions is triggered and the machine learning prediction disagrees with the rule, the application overrides the model prediction.

The application then updates the displayed probability distribution to reflect the overridden prediction.

> **Important:** These rules are project-specific logic and are not clinically validated. The fixed probability assigned during an override should not be interpreted as a medically validated probability.

---

## 🌐 Web Application

The project provides a Flask-based web interface where users can enter the five input parameters and receive a prediction.

### Backend

The Flask backend:

- Loads the trained XGBoost model
- Loads the saved StandardScaler
- Loads the class-name mapping
- Receives user input through a POST request
- Converts the input values to numeric format
- Applies the saved feature scaling
- Generates the model prediction
- Calculates prediction probabilities
- Applies the rule-based TSH override when applicable
- Sends prediction results to the result page
- Handles unexpected errors using a custom error page

### Frontend

The frontend includes:

- Responsive input form
- Tailwind CSS styling
- Prediction result display
- Prediction confidence display
- Probability breakdown
- Input-value summary
- New prediction option
- Downloadable text prediction report
- Custom error page

---

## 📁 Project Structure

```text
Thyroid_Disease_Prediction/
│
├── data/
│   └── thyroid_dataset.csv
│
├── templates/
│   ├── error.html
│   ├── index.html
│   └── result.html
│
├── .gitignore
├── app.py
├── best_thyroid_model.pkl
├── Procfile
├── README.md
├── requirements.txt
├── runtime.txt
├── thyroid_class_names.pkl
├── thyroid_dataset.csv
├── thyroid_scaler.pkl
└── Thyroid.ipynb
```

> **Note:** The repository currently contains `thyroid_dataset.csv` in both the project root and the `data/` directory. The training notebook currently loads the root-level `thyroid_dataset.csv`.

---

## 💾 Saved Model Files

The following serialized files are included in the repository:

| File | Purpose |
|---|---|
| `best_thyroid_model.pkl` | Trained XGBoost classifier |
| `thyroid_scaler.pkl` | Fitted StandardScaler |
| `thyroid_class_names.pkl` | Class-label mapping |

The scaler is saved separately so that new user inputs are transformed using the same scaling procedure used during model training.

---

## 🔬 Model Development

The machine learning development process was performed in:

`Thyroid.ipynb`

The notebook contains:

- Dataset loading
- Data cleaning
- Feature preparation
- Target encoding
- Train-test splitting
- Feature scaling
- SMOTE-based class balancing
- Model training
- Model comparison
- Classification reports
- Confusion matrices
- Final model selection
- Model persistence

---

## 🛠️ Technologies Used

**Programming Language**
- Python 3.12

**Machine Learning**
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

**Data Processing**
- Pandas
- NumPy

**Data Visualization**
- Matplotlib
- Seaborn

**Web Development**
- Flask
- HTML
- Tailwind CSS
- JavaScript
- Jinja Templates

**Model Persistence**
- Pickle

**Deployment**
- Gunicorn

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
```

> Replace the repository URL with your actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Flask Application

```bash
python app.py
```

### 6. Open the Application

Open the following address in your browser:

```text
http://localhost:5000
```

---

## 📋 Requirements

The project dependencies are specified in `requirements.txt`.

The main dependencies include:

- Flask
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Seaborn
- Gunicorn

---

## 🚀 Deployment

The application is configured for deployment using Gunicorn through the included `Procfile`.

The application uses the following command:

```text
web: gunicorn app:app
```

The Flask application also reads the deployment port from the `PORT` environment variable and defaults to port 5000 for local execution.

---

## ⚠️ Limitations

- The dataset is relatively small.
- Model evaluation is based on a single train-test split.
- The model has not been clinically validated.
- The rule-based TSH override is a project-specific implementation and is not a validated clinical decision rule.
- Model probabilities should not be interpreted as medical certainty.
- The system should not be used to diagnose or treat a medical condition.

---

## 🔮 Future Improvements

Potential improvements include:

- Cross-validation for more robust model evaluation
- Hyperparameter tuning
- Evaluation on a larger and more diverse dataset
- Probability calibration
- Automated model selection based on validation performance
- Improved input validation
- Additional model explainability techniques such as SHAP
- Containerized deployment
- Automated testing
- Model monitoring

---

## ⚕️ Medical Disclaimer

This project is intended for educational and research purposes only.

It is not a substitute for professional medical advice, diagnosis, or treatment.

The predictions generated by this application should not be used to make medical decisions.

Always consult a qualified healthcare professional for medical evaluation, diagnosis, and treatment.

---

## 👨‍💻 Project Purpose

This project demonstrates an end-to-end machine learning workflow, from data preprocessing and class balancing to model comparison, evaluation, model persistence, and Flask-based web deployment.

It was developed as a practical machine learning project to explore how a classification model can be integrated into a functional web application.

---

## 📌 Future Development

The project can be further improved by expanding the dataset, using more robust validation techniques, improving model calibration and explainability, and adding automated testing and monitoring.

---

**Built as a machine learning project for educational and research purposes.**
