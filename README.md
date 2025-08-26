# 🧪 Thyroid Disease Prediction System

A comprehensive machine learning-powered web application for predicting thyroid diseases (Normal, Hyperthyroid, Hypothyroid) based on key thyroid function tests. Built with Flask, advanced ML algorithms, and a modern responsive UI.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/xgboost-v1.6+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Project Overview

This project implements an end-to-end machine learning solution for thyroid disease classification using clinical thyroid function test parameters. The system achieves **97.67% accuracy** with the XGBoost classifier and includes intelligent rule-based overrides for extreme TSH values to enhance clinical reliability.

### Key Features
- 🤖 **Advanced ML Pipeline**: XGBoost classifier with SMOTE balancing and cross-validation
- 🎨 **Modern Web Interface**: Responsive design with Tailwind CSS and glassmorphism effects
- ⚡ **Real-time Predictions**: Instant thyroid condition assessment
- 📊 **Detailed Analytics**: Probability breakdown and confidence scores
- 🩺 **Clinical Rule Integration**: Smart overrides for extreme TSH values
- 📁 **Report Generation**: Downloadable prediction reports
- 🔒 **Error Handling**: Comprehensive validation and error management

## 🏥 Medical Background

The system predicts three thyroid conditions based on five key biomarkers:

### Conditions Classified:
- **Normal**: Healthy thyroid function
- **Hyperthyroid**: Overactive thyroid (low TSH < 0.2 μIU/mL)
- **Hypothyroid**: Underactive thyroid (high TSH > 10.0 μIU/mL)

### Input Parameters:
- **T3 Resin Uptake Test (%)**: Normal range 25-35%
- **Total Serum Thyroxin (TT4)**: Normal range 60-140 µg/dL
- **Total Serum Triiodothyronine (T3)**: Normal range 0.8-2.8 ng/mL
- **Basal TSH**: Normal range 0.4-4.0 μIU/mL
- **Max TSH Difference**: Post-stimulation TSH change

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/thyroid-disease-prediction.git
   cd thyroid-disease-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if needed)
   ```bash
   python thyroid_classification.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the web interface**
   ```
   Open your browser and navigate to: http://localhost:5000
   ```

## 📁 Project Structure

```
thyroid-disease-prediction/
│
├── app.py                          # Flask web application
├── thyroid_classification.py       # ML training pipeline
├── thyroid_dataset.csv             # Training dataset
│
├── templates/                      # HTML templates
│   ├── index.html                  # Main input form
│   ├── result.html                 # Prediction results page
│   └── error.html                  # Error handling page
│
├── model_files/                    # Saved model artifacts
│   ├── best_thyroid_model.pkl      # Trained XGBoost model
│   ├── thyroid_scaler.pkl          # Feature scaler
│   └── thyroid_class_names.pkl     # Class label mappings
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                     # Git ignore rules
```

## 🧠 Machine Learning Pipeline

### Data Processing
- **Dataset**: 215 samples with 6 features
- **Class Distribution**: Normal (150), Hyperthyroid (35), Hypothyroid (30)
- **Preprocessing**: StandardScaler normalization
- **Balancing**: SMOTE oversampling for class imbalance
- **Split**: 80/20 train-test with stratification

### Model Comparison
| Model | Accuracy | CV Score | Status |
|-------|----------|----------|--------|
| **XGBoost** | **97.67%** | **97.78%** | 🏆 **Selected** |
| Logistic Regression | 100.00% | 98.61% | Overfitting risk |
| SVM | 100.00% | 98.06% | Overfitting risk |
| Random Forest | 97.67% | 97.22% | Good alternative |
| Decision Tree | 95.35% | 96.11% | Lower performance |

### Model Features
- **Algorithm**: XGBoost Classifier
- **Cross-validation**: 5-fold stratified
- **Class weights**: Balanced for minority classes
- **Hyperparameters**: Optimized for medical data
- **Validation**: Comprehensive metrics and confusion matrices

## 🌐 Web Application

### Frontend Features
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **User Experience**: Intuitive form with input validation and helpful hints
- **Visual Feedback**: Color-coded predictions and progress bars
- **Modern UI**: Glassmorphism effects and smooth animations

### Backend Architecture
- **Framework**: Flask with modular design
- **Model Loading**: Efficient pickle-based model persistence
- **Error Handling**: Comprehensive exception management
- **Rule Integration**: Clinical override logic for extreme values

### API Endpoints
- `GET /`: Main input form
- `POST /predict`: Prediction processing
- Custom error handlers for 404/500 errors

## 🎯 Model Performance

### Test Set Results
```
Overall Accuracy: 97.67%
Cross-validation: 97.78% ± 1.42%

Classification Report:
                 precision    recall  f1-score   support
Normal              0.97      1.00      0.98        30
Hyperthyroid        1.00      0.86      0.92         7
Hypothyroid         1.00      1.00      1.00         6

accuracy                                0.98        43
macro avg           0.99      0.95      0.97        43
weighted avg        0.98      0.98      0.98        43
```

### Clinical Rule Integration
The system includes intelligent overrides:
- **TSH < 0.2**: Automatic Hyperthyroid classification (100% confidence)
- **TSH > 10.0**: Automatic Hypothyroid classification (100% confidence)

## 🔧 Technical Implementation

### Dependencies
```python
Flask==2.3.2
scikit-learn==1.3.0
xgboost==1.7.6
pandas==2.0.3
numpy==1.24.3
imbalanced-learn==0.11.0
matplotlib==3.7.2
seaborn==0.12.2
```

### Key Technologies
- **Machine Learning**: Scikit-learn, XGBoost, SMOTE
- **Web Framework**: Flask
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Model Persistence**: Pickle serialization

## 📊 Usage Example

```python
# Example input values
input_data = {
    'T3_resin_uptake': 28.5,
    'total_serum_thyroxin': 110.0,
    'total_serum_t3': 2.1,
    'basal_tsh': 2.5,
    'max_diff_tsh': 1.8
}

# Expected output
{
    'prediction': 'Normal',
    'confidence': '95.67%',
    'probabilities': {
        'Normal': '95.67%',
        'Hyperthyroid': '2.31%',
        'Hypothyroid': '2.02%'
    }
}
```

## 🚀 Deployment

### Local Development
```bash
python app.py
# Access: http://localhost:5000
```

### Production Deployment
The application is configured for deployment on platforms like:
- **Heroku**: Use the included Procfile
- **AWS EC2**: Deploy with gunicorn
- **Docker**: Containerized deployment ready
- **Railway/Render**: Direct deployment from Git

### Environment Variables
```bash
PORT=5000  # Optional: defaults to 5000
DEBUG=False  # Set to False in production
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure cross-validation scores remain stable

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical concerns.

## 🙏 Acknowledgments

- **Dataset**: Clinical thyroid function test data
- **Libraries**: Scikit-learn, XGBoost, Flask communities
- **UI Framework**: Tailwind CSS for modern design
- **Medical References**: Thyroid function testing guidelines

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/yourusername/thyroid-disease-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/thyroid-disease-prediction/discussions)
- **Email**: your.email@example.com

---

**Built with ❤️ for better healthcare through AI**

*Last updated: August 2025*
