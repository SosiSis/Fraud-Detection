# 🛡️ Advanced Fraud Detection System

## 10 Academy: Artificial Intelligence Mastery - Week 8&9 Challenge

A comprehensive fraud detection system for e-commerce and credit card transactions, featuring advanced machine learning models, explainable AI, REST API deployment, and interactive dashboards.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.0+-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![MLflow](https://img.shields.io/badge/mlflow-tracking-orange.svg)
![Dash](https://img.shields.io/badge/dash-dashboard-red.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Dashboard](#dashboard)
- [Model Performance](#model-performance)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a state-of-the-art fraud detection system that addresses the unique challenges of both e-commerce and credit card transaction fraud. The system combines traditional machine learning algorithms with deep learning models, provides explainable AI capabilities, and offers production-ready deployment options.

### Business Impact

- **Enhanced Security**: Accurately identifies fraudulent transactions in real-time
- **Cost Reduction**: Minimizes false positives and financial losses
- **Compliance**: Provides explainable predictions for regulatory requirements
- **Scalability**: Cloud-ready architecture supports high-volume processing

## ✨ Features

### 🤖 Machine Learning Models
- **Traditional ML**: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting
- **Deep Learning**: Multi-Layer Perceptron (MLP), Convolutional Neural Networks (CNN), LSTM
- **Model Comparison**: Comprehensive evaluation and selection
- **MLOps Integration**: Experiment tracking with MLflow

### 🔍 Explainable AI
- **SHAP Analysis**: Global and local feature importance
- **LIME Explanations**: Individual prediction interpretability
- **Model Comparison**: Understanding different model behaviors
- **Business Insights**: Actionable recommendations

### 🚀 Production Deployment
- **REST API**: Flask-based API with comprehensive endpoints
- **Docker Support**: Containerized deployment
- **Health Monitoring**: API health checks and logging
- **Batch Processing**: Support for bulk predictions

### 📊 Interactive Dashboard
- **Real-time Monitoring**: Live fraud detection metrics
- **Geographic Analysis**: Location-based fraud patterns
- **Time Series Analysis**: Temporal fraud trends
- **Device Analysis**: Browser and device risk patterns
- **Live Predictions**: Interactive fraud prediction interface

## 📁 Project Structure

```
fraud-detection-system/
├── data/                           # Dataset storage
│   ├── Fraud_Data.csv             # E-commerce fraud data
│   ├── IpAddress_to_Country.csv   # IP geolocation mapping
│   └── creditcard.csv             # Credit card fraud data
├── notebooks/                      # Jupyter notebooks
│   ├── task1_data_analysis_preprocessing.ipynb
│   ├── task2_model_building_training.ipynb
│   └── task3_model_explainability.ipynb
├── models/                         # Trained models
│   ├── best_fraud_model.pkl
│   ├── best_credit_model.pkl
│   ├── fraud_scaler.pkl
│   └── credit_scaler.pkl
├── explainability_results/        # SHAP/LIME outputs
├── logs/                          # Application logs
├── serve_model.py                 # Flask API application
├── dashboard_app.py               # Dash dashboard
├── test_api.py                    # API testing suite
├── train_models_for_api.py        # Model training script
├── create_sample_data.py          # Data generation script
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Multi-container setup
├── deploy.sh                      # Deployment script
└── README.md                      # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized deployment)
- 8GB+ RAM (recommended for deep learning models)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/SosiSis/Fraud-Detection
   cd Fraud-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data** (if datasets not provided)
   ```bash
   python create_sample_data.py
   ```

4. **Train models**
   ```bash
   python train_models_for_api.py
   ```

5. **Start the API**
   ```bash
   python serve_model.py
   ```

6. **Launch dashboard** (in another terminal)
   ```bash
   python dashboard_app.py
   ```

### Docker Deployment

For production deployment with Docker:

```bash
# Quick deployment
./deploy.sh

# Or manual Docker commands
docker buildx build -t fraud-detection-api .
docker run -p 5000:5000 fraud-detection-api
```

## 📖 Usage

### Running Jupyter Notebooks

Execute the notebooks in order to understand the complete workflow:

1. **Task 1 - Data Analysis & Preprocessing**
   ```bash
   jupyter notebook notebooks/task1_data_analysis_preprocessing.ipynb
   ```

2. **Task 2 - Model Building & Training**
   ```bash
   jupyter notebook notebooks/task2_model_building_training.ipynb
   ```

3. **Task 3 - Model Explainability**
   ```bash
   jupyter notebook notebooks/task3_model_explainability.ipynb
   ```

### API Usage Examples

```bash
# Health check
curl http://localhost:5000/health

# Fraud prediction
curl -X POST http://localhost:5000/predict/fraud \
  -H "Content-Type: application/json" \
  -d '{
    "purchase_value": 150.0,
    "age": 35,
    "hour_of_day": 14,
    "day_of_week": 2,
    "source_encoded": 1,
    "browser_encoded": 0,
    "sex_encoded": 1
  }'

# Get explanation
curl -X POST http://localhost:5000/explain/fraud \
  -H "Content-Type: application/json" \
  -d '{...same data...}'
```

## 🔌 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health status |
| GET | `/` | API documentation |
| POST | `/predict/fraud` | E-commerce fraud prediction |
| POST | `/predict/credit` | Credit card fraud prediction |
| POST | `/explain/fraud` | SHAP explanation for fraud prediction |
| POST | `/explain/credit` | SHAP explanation for credit prediction |
| POST | `/batch/fraud` | Batch fraud prediction |
| POST | `/batch/credit` | Batch credit prediction |

### Request/Response Examples

#### Fraud Prediction Request
```json
{
  "purchase_value": 150.0,
  "age": 35,
  "hour_of_day": 14,
  "day_of_week": 2,
  "source_encoded": 1,
  "browser_encoded": 0,
  "sex_encoded": 1
}
```

#### Fraud Prediction Response
```json
{
  "prediction": 0,
  "probability": {
    "non_fraud": 0.95,
    "fraud": 0.05
  },
  "risk_score": 0.05,
  "risk_level": "LOW",
  "timestamp": "2025-02-18T10:30:00",
  "model_version": "1.0"
}
```

## 📊 Dashboard

The interactive dashboard provides comprehensive fraud monitoring capabilities:

### Features
- **Real-time Metrics**: Live fraud detection statistics
- **Geographic Analysis**: World map showing fraud patterns by location
- **Temporal Analysis**: Time-based fraud trends and patterns
- **Device Intelligence**: Browser and device risk analysis
- **Live Prediction**: Interactive fraud prediction interface

### Access
- **URL**: http://localhost:8050
- **Requirements**: API must be running on port 5000/5001

### Screenshots

*Dashboard screenshots would be included here in a real deployment*

## 📈 Model Performance

### E-commerce Fraud Detection
- **Best Model**: Random Forest
- **Accuracy**: 97.82%
- **ROC AUC**: 84.92%
- **F1 Score**: 0.76

### Credit Card Fraud Detection
- **Best Model**: Random Forest
- **Accuracy**: 99.93%
- **ROC AUC**: 86.21%
- **F1 Score**: 0.82

### Model Comparison
| Model | Fraud Accuracy | Credit Accuracy | Fraud ROC AUC | Credit ROC AUC |
|-------|---------------|----------------|---------------|----------------|
| Logistic Regression | 90.57% | 99.91% | 50.00% | 78.05% |
| Decision Tree | 90.75% | 99.91% | 75.75% | 90.79% |
| Random Forest | 97.82% | 99.93% | 84.92% | 86.21% |
| Gradient Boosting | 95.23% | 99.92% | 82.15% | 85.43% |
| MLP | 91.45% | 99.89% | 68.32% | 79.87% |

## 🐳 Docker Deployment

### Single Container
```bash
# Build image
docker buildx build -t fraud-detection-api .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  fraud-detection-api
```

### Multi-Container with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment
```bash
# Deploy with nginx reverse proxy
docker-compose --profile production up -d
```

## 🛠️ Development

### Running Tests
```bash
# API tests
python test_api.py

# Model tests
python -m pytest tests/ -v
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Adding New Models

1. Implement model in `notebooks/task2_model_building_training.ipynb`
2. Save model artifacts to `models/` directory
3. Update `serve_model.py` to load new model
4. Add prediction endpoint
5. Update tests and documentation

## 📝 Key Insights & Recommendations

### Business Insights
1. **Feature Importance**: Purchase value and transaction timing are key fraud indicators
2. **Geographic Patterns**: Certain regions show higher fraud rates requiring targeted monitoring
3. **Device Analysis**: Browser and device combinations create distinct risk profiles
4. **Temporal Patterns**: Late-night and weekend transactions require enhanced scrutiny

### Technical Recommendations
1. **Model Ensemble**: Combine multiple models for improved accuracy
2. **Real-time Processing**: Implement streaming for immediate fraud detection
3. **Feature Engineering**: Develop more sophisticated temporal and behavioral features
4. **Continuous Learning**: Implement online learning for model adaptation

### Operational Recommendations
1. **Threshold Tuning**: Regularly adjust risk thresholds based on business requirements
2. **A/B Testing**: Continuously test new models against production systems
3. **Monitoring**: Implement comprehensive model performance monitoring
4. **Feedback Loop**: Establish mechanisms for incorporating fraud investigator feedback

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **10 Academy** for providing the challenge framework
- **Adey Innovations Inc.** for the business context
- **Open Source Community** for the excellent libraries used
- **Contributors** who helped improve this project

## 📞 Support

For questions, issues, or contributions:

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: [sosinasisay29@gmail.com]

---

**Built with ❤️ for the 10 Academy AI Mastery Challenge**

*This project demonstrates advanced fraud detection capabilities using state-of-the-art machine learning techniques, explainable AI, and production-ready deployment practices.*
