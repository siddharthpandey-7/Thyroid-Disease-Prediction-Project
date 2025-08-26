# 🧪 Thyroid Disease Prediction

A machine learning web application that predicts thyroid disease based on medical test results using Flask and scikit-learn.

## 📋 Features

- **Real-time Prediction**: Get instant thyroid disease predictions
- **Multiple Disease Types**: Predicts Normal, Hyperthyroid, and Hypothyroid conditions
- **Confidence Scores**: Shows prediction confidence and probability breakdown
- **User-friendly Interface**: Clean, responsive web interface
- **Input Validation**: Ensures valid medical test values

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

## 📊 Input Parameters

The application requires 5 medical test values:

1. **T3 Resin Uptake Test (%)** - Normal range: 25–35
2. **Total Serum Thyroxin (TT4)** - Normal range: 60–140 µg/dL
3. **Total Serum Triiodothyronine (T3)** - Normal range: 0.8–2.8 ng/mL
4. **Basal TSH** - Normal range: 0.4–4.0 μIU/mL
5. **Max TSH difference after thyrotropin injection** - Measured after stimulation

## 🎯 Prediction Results

The application provides:
- **Primary Prediction**: Normal, Hyperthyroid, or Hypothyroid
- **Confidence Score**: Percentage confidence in the prediction
- **Probability Breakdown**: Individual probabilities for each condition
- **Input Summary**: Review of your entered values

## 📁 Project Structure

```
Thyroid_Disease_Prediction/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/
│   └── thyroid_dataset.csv    # Training dataset
├── templates/
│   ├── index.html        # Main form page
│   ├── result.html       # Results display page
│   └── error.html        # Error page
├── notebooks/
│   └── Thyroid2.ipynb   # Jupyter notebook with ML analysis
└── Model files:
    ├── best_thyroid_model.pkl      # Trained ML model
    ├── thyroid_scaler.pkl          # Data scaler
    └── thyroid_class_names.pkl     # Class name mappings
```

## 🔬 Machine Learning Details

- **Dataset**: 215 samples with 5 features
- **Classes**: Normal (150), Hyperthyroid (35), Hypothyroid (30)
- **Model**: Optimized ensemble model with SMOTE balancing
- **Features**: Standardized using StandardScaler
- **Validation**: Cross-validation and hyperparameter tuning

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn, numpy, pandas
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: StandardScaler, SMOTE for balancing

## ⚠️ Important Notes

- This is a **demonstration tool** and should not replace professional medical diagnosis
- Always consult healthcare professionals for medical decisions
- The model is trained on a specific dataset and may not generalize to all populations
- Input validation ensures only positive numeric values are accepted

## 🐛 Troubleshooting

**If the form doesn't show results:**
1. Make sure all model files are in the same directory as `app.py`
2. Check that all dependencies are installed: `pip install -r requirements.txt`
3. Ensure you're entering valid numeric values
4. Check the console for any error messages

**Common Issues:**
- **"Module not found"**: Run `pip install -r requirements.txt`
- **"Model files not found"**: Ensure all `.pkl` files are in the project root
- **Form resets**: This is normal behavior - the form clears after submission

## 📈 Model Performance

The trained model achieves:
- High accuracy on the thyroid dataset
- Balanced performance across all three classes
- Robust predictions with confidence scoring

## 🤝 Contributing

Feel free to improve the application by:
- Adding more features
- Improving the UI/UX
- Enhancing the machine learning model
- Adding more validation rules

## 📄 License

This project is for educational and demonstration purposes.

---

**Disclaimer**: This application is for educational purposes only and should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.



