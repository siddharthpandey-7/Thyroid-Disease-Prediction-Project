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
