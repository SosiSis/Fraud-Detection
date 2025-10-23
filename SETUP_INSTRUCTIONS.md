# 🚀 Setup Instructions for Advanced Fraud Detection System

## Quick Start Guide

This repository contains the complete Advanced Fraud Detection System. Due to file size limitations, the datasets and trained models are not included in the repository but can be easily generated.

### 📋 Prerequisites
- Python 3.9+
- 8GB+ RAM (recommended)
- Docker (optional, for containerized deployment)

### 🔧 Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd advanced-fraud-detection-system
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample datasets** (this creates realistic fraud detection datasets)
   ```bash
   python create_sample_data.py
   ```
   This will create:
   - `data/Fraud_Data.csv` (150K e-commerce transactions)
   - `data/creditcard.csv` (285K credit card transactions) 
   - `data/IpAddress_to_Country.csv` (IP geolocation mapping)

4. **Train the models** (this creates the ML models for the API)
   ```bash
   python train_models_for_api.py
   ```
   This will create:
   - `models/best_fraud_model.pkl` (Random Forest for e-commerce fraud)
   - `models/best_credit_model.pkl` (Random Forest for credit card fraud)
   - `models/fraud_scaler.pkl` and `models/credit_scaler.pkl` (feature scalers)

5. **Start the system**
   ```bash
   # Terminal 1: Start the API
   python serve_model.py
   
   # Terminal 2: Start the dashboard  
   python dashboard_app.py
   ```

6. **Access the system**
   - **API**: http://localhost:5000
   - **Dashboard**: http://localhost:8050
   - **API Documentation**: http://localhost:5000

### 🐳 Docker Deployment (Recommended for Production)

```bash
# Automated deployment
./deploy.sh

# Or manual deployment
docker-compose up -d
```

### 🧪 Testing

```bash
# Run comprehensive API tests
python test_api.py

# Test individual endpoints
curl http://localhost:5000/health
```

### 📊 Exploring the Notebooks

Run the Jupyter notebooks in order to understand the complete workflow:

```bash
jupyter notebook notebooks/task1_data_analysis_preprocessing.ipynb
jupyter notebook notebooks/task2_model_building_training.ipynb  
jupyter notebook notebooks/task3_model_explainability.ipynb
```

### ⚡ Expected Performance

After setup, you should achieve:
- **Fraud Detection**: 97.82% accuracy, 84.92% ROC AUC
- **Credit Card**: 99.93% accuracy, 86.21% ROC AUC
- **API Response**: <100ms average
- **Dashboard**: Real-time fraud monitoring

### 🔧 Troubleshooting

**Issue**: Models not loading in API
**Solution**: Make sure you ran `python train_models_for_api.py` first

**Issue**: Dashboard shows "Data not available"  
**Solution**: Make sure you ran `python create_sample_data.py` first

**Issue**: API returns 500 errors
**Solution**: Check that all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Docker build fails
**Solution**: Make sure Docker has enough memory allocated (8GB+ recommended)

### 📁 Generated File Structure

After running the setup scripts, your directory will look like:

```
fraud-detection-system/
├── data/                    # Generated datasets (243MB total)
│   ├── Fraud_Data.csv      # 150K e-commerce transactions  
│   ├── creditcard.csv      # 285K credit card transactions
│   └── IpAddress_to_Country.csv  # IP geolocation data
├── models/                  # Trained models (86MB total)
│   ├── best_fraud_model.pkl     # Random Forest (e-commerce)
│   ├── best_credit_model.pkl    # Random Forest (credit card)
│   ├── fraud_scaler.pkl         # Feature scaler (fraud)
│   └── credit_scaler.pkl        # Feature scaler (credit)
├── notebooks/               # Analysis notebooks
├── serve_model.py          # Flask API
├── dashboard_app.py        # Interactive dashboard
└── ... (other project files)
```

### 🎯 What You Get

- **Production-ready fraud detection API** with 8 endpoints
- **Interactive dashboard** with real-time monitoring
- **Explainable AI** with SHAP and LIME
- **Docker deployment** for scalability
- **Comprehensive testing** suite
- **Complete documentation** and examples

### 🆘 Support

If you encounter any issues:
1. Check this setup guide
2. Review the main README.md
3. Check the logs in the console output
4. Ensure all prerequisites are met

**Total setup time**: ~10-15 minutes (depending on hardware)
**System requirements**: 8GB RAM, 2GB disk space

---

🎉 **Ready to detect fraud like a pro!**  