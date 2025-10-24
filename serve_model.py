#!/usr/bin/env python3
"""
Advanced Fraud Detection API
Flask application for serving fraud detection models
"""

import os
import sys
import logging
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import json

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import shap

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduced log level for production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Only stdout in production
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for models and scalers
fraud_model = None
credit_model = None
fraud_scaler = None
credit_scaler = None
fraud_explainer = None
credit_explainer = None

# Feature names for reference
FRAUD_FEATURES = [
    'purchase_value', 'age', 'hour_of_day', 'day_of_week', 
    'source_encoded', 'browser_encoded', 'sex_encoded'
]

CREDIT_FEATURES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

def load_models():
    """Load trained models and scalers"""
    global fraud_model, credit_model, fraud_scaler, credit_scaler
    global fraud_explainer, credit_explainer
    
    try:
        # Try to load pre-trained models
        if os.path.exists('models/best_fraud_model.pkl'):
            fraud_model = joblib.load('models/best_fraud_model.pkl')
            fraud_scaler = joblib.load('models/fraud_scaler.pkl')
            logger.info("Loaded pre-trained fraud model")
        else:
            # Train a simple model if pre-trained not available
            logger.warning("Pre-trained fraud model not found, training simple model...")
            train_simple_fraud_model()
            
        if os.path.exists('models/best_credit_model.pkl'):
            credit_model = joblib.load('models/best_credit_model.pkl')
            credit_scaler = joblib.load('models/credit_scaler.pkl')
            logger.info("Loaded pre-trained credit model")
        else:
            logger.warning("Pre-trained credit model not found, training simple model...")
            train_simple_credit_model()
            
        # Initialize SHAP explainers
        if fraud_model is not None:
            try:
                # Use TreeExplainer for tree-based models, otherwise fallback to a more general explainer
                model_type_name = str(type(fraud_model))
                if 'RandomForest' in model_type_name or 'DecisionTree' in model_type_name or 'XGB' in model_type_name or 'LGBM' in model_type_name:
                    fraud_explainer = shap.TreeExplainer(fraud_model)
                    logger.info("Initialized SHAP TreeExplainer for fraud model")
                else:
                    # KernelExplainer is a model-agnostic explainer. It requires a background dataset.
                    # For simplicity, we use a zeroed-out array. For better accuracy, a sample of the training data is recommended.
                    fraud_explainer = shap.KernelExplainer(fraud_model.predict_proba, np.zeros((1, len(FRAUD_FEATURES))))
                    logger.info(f"Initialized SHAP KernelExplainer for fraud model of type {model_type_name}")
            except Exception as e:
                logger.warning(f"Could not initialize fraud SHAP explainer: {e}")
                
        if credit_model is not None:
            try:
                model_type_name = str(type(credit_model))
                if 'RandomForest' in model_type_name or 'DecisionTree' in model_type_name or 'XGB' in model_type_name or 'LGBM' in model_type_name:
                    credit_explainer = shap.TreeExplainer(credit_model)
                    logger.info("Initialized SHAP TreeExplainer for credit model")
                else:
                    credit_explainer = shap.KernelExplainer(credit_model.predict_proba, np.zeros((1, len(CREDIT_FEATURES))))
                    logger.info(f"Initialized SHAP KernelExplainer for credit model of type {model_type_name}")
            except Exception as e:
                logger.warning(f"Could not initialize credit SHAP explainer: {e}")
                
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error(traceback.format_exc())

def train_simple_fraud_model():
    """Train a simple fraud model if pre-trained not available"""
    global fraud_model, fraud_scaler
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Load sample data
        if os.path.exists('data/Fraud_Data.csv'):
            data = pd.read_csv('data/Fraud_Data.csv')
            
            # Basic preprocessing
            data['signup_time'] = pd.to_datetime(data['signup_time'])
            data['purchase_time'] = pd.to_datetime(data['purchase_time'])
            data['hour_of_day'] = data['purchase_time'].dt.hour
            data['day_of_week'] = data['purchase_time'].dt.dayofweek
            
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in ['source', 'browser', 'sex']:
                data[f'{col}_encoded'] = le.fit_transform(data[col])
            
            X = data[FRAUD_FEATURES]
            y = data['class']
            
            # Train model
            fraud_scaler = StandardScaler()
            X_scaled = fraud_scaler.fit_transform(X)
            
            fraud_model = RandomForestClassifier(n_estimators=50, random_state=42)
            fraud_model.fit(X_scaled, y)
            
            logger.info("Trained simple fraud model successfully")
            
        else:
            logger.error("No fraud data available for training")
            
    except Exception as e:
        logger.error(f"Error training simple fraud model: {e}")

def train_simple_credit_model():
    """Train a simple credit model if pre-trained not available"""
    global credit_model, credit_scaler
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Load sample data
        if os.path.exists('data/creditcard.csv'):
            data = pd.read_csv('data/creditcard.csv')
            
            X = data.drop(columns=['Class'])
            y = data['Class']
            
            # Train model
            credit_scaler = StandardScaler()
            X_scaled = credit_scaler.fit_transform(X)
            
            credit_model = RandomForestClassifier(n_estimators=50, random_state=42)
            credit_model.fit(X_scaled, y)
            
            logger.info("Trained simple credit model successfully")
            
        else:
            logger.error("No credit data available for training")
            
    except Exception as e:
        logger.error(f"Error training simple credit model: {e}")

def validate_fraud_input(data):
    """Validate fraud detection input data"""
    required_fields = FRAUD_FEATURES
    
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    # Validate data types and ranges
    try:
        purchase_value = float(data['purchase_value'])
        age = int(data['age'])
        hour_of_day = int(data['hour_of_day'])
        day_of_week = int(data['day_of_week'])
        
        if not (0 <= hour_of_day <= 23):
            return False, "hour_of_day must be between 0 and 23"
        if not (0 <= day_of_week <= 6):
            return False, "day_of_week must be between 0 and 6"
        if not (18 <= age <= 100):
            return False, "age must be between 18 and 100"
        if purchase_value < 0:
            return False, "purchase_value must be non-negative"
            
    except (ValueError, TypeError):
        return False, "Invalid data types in input"
    
    return True, "Valid"

def validate_credit_input(data):
    """Validate credit card input data"""
    required_fields = CREDIT_FEATURES
    
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    # Validate data types
    try:
        for field in required_fields:
            float(data[field])
    except (ValueError, TypeError):
        return False, "All fields must be numeric"
    
    return True, "Valid"

@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Fraud Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; }
            .endpoint { background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #2196F3; font-weight: bold; }
            pre { background-color: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto; }
            .status { color: green; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>  Advanced Fraud Detection API</h1>
            <p>Professional fraud detection system for e-commerce and credit card transactions</p>
            <p class="status">Status: {{ status }}</p>
        </div>
        
        <h2>📋 Available Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Check API health status</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict/fraud</h3>
            <p>Predict fraud for e-commerce transactions</p>
            <h4>Required Fields:</h4>
            <pre>{{ fraud_fields }}</pre>
            <h4>Example Request:</h4>
            <pre>{{ fraud_example }}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict/credit</h3>
            <p>Predict fraud for credit card transactions</p>
            <h4>Required Fields:</h4>
            <pre>{{ credit_fields }}</pre>
        </
            <p>Batch prediction for multiple fraud transactions</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /batch/credit</h3>
            <p>Batch prediction for multiple credit card transactions</p>
        </div>
        
        <h2>📊 Model Information</h2>
        <p><strong>Fraud Model:</strong> {{ fraud_model_info }}</p>
        <p><strong>Credit Model:</strong> {{ credit_model_info }}</p>
        
        <h2>🔗 Usage Examples</h2>
        <pre>
# Test fraud prediction
curl -X POST http://localhost:5000/predict/fraud \\
  -H "Content-Type: application/json" \\
  -d '{{ fraud_example }}'

# Check health
curl http://localhost:5000/health
        </pre>
    </body>
    </html>
    """
    
    fraud_example = json.dumps({
        "purchase_value": 150.0,
        "age": 35,
        "hour_of_day": 14,
        "day_of_week": 2,
        "source_encoded": 1,
        "browser_encoded": 0,
        "sex_encoded": 1
    }, indent=2)
    
    return render_template_string(html_template,
        status="✅ Online" if fraud_model and credit_model else "⚠️ Models Loading",
        fraud_fields=json.dumps(FRAUD_FEATURES, indent=2),
        credit_fields="Time, V1-V28, Amount (30 total fields)",
        fraud_example=fraud_example,
        fraud_model_info=str(type(fraud_model).__name__) if fraud_model else "Not loaded",
        credit_model_info=str(type(credit_model).__name__) if credit_model else "Not loaded"
    )

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models': {
                'fraud_model': fraud_model is not None,
                'credit_model': credit_model is not None,
                'fraud_scaler': fraud_scaler is not None,
                'credit_scaler': credit_scaler is not None
            },
            'explainers': {
                'fraud_explainer': fraud_explainer is not None,
                'credit_explainer': credit_explainer is not None
            }
        }
        
        logger.info("Health check requested")
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/predict/fraud', methods=['POST'])
def predict_fraud():
    """Predict fraud for e-commerce transaction"""
    try:
        # Log request
        logger.info(f"Fraud prediction request from {request.remote_addr}")
        
        if fraud_model is None or fraud_scaler is None:
            return jsonify({'error': 'Fraud model not available'}), 500
        
        # Get and validate input
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        is_valid, message = validate_fraud_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Prepare features
        features = np.array([[data[field] for field in FRAUD_FEATURES]])
        features_scaled = fraud_scaler.transform(features)
        
        # Make prediction
        prediction = fraud_model.predict(features_scaled)[0]
        probability = fraud_model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': int(prediction),
            'probability': {
                'non_fraud': float(probability[0]),
                'fraud': float(probability[1])
            },
            'risk_score': float(probability[1]),
            'risk_level': 'HIGH' if probability[1] > 0.7 else 'MEDIUM' if probability[1] > 0.3 else 'LOW',
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        
        # Log prediction
        logger.info(f"Fraud prediction: {prediction}, Risk score: {probability[1]:.3f}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in fraud prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict/credit', methods=['POST'])
def predict_credit():
    """Predict fraud for credit card transaction"""
    try:
        # Log request
        logger.info(f"Credit prediction request from {request.remote_addr}")
        
        if credit_model is None or credit_scaler is None:
            return jsonify({'error': 'Credit model not available'}), 500
        
        # Get and validate input
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        is_valid, message = validate_credit_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Prepare features
        features = np.array([[data[field] for field in CREDIT_FEATURES]])
        features_scaled = credit_scaler.transform(features)
        
        # Make prediction
        prediction = credit_model.predict(features_scaled)[0]
        probability = credit_model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': int(prediction),
            'probability': {
                'non_fraud': float(probability[0]),
                'fraud': float(probability[1])
            },
            'risk_score': float(probability[1]),
            'risk_level': 'HIGH' if probability[1] > 0.7 else 'MEDIUM' if probability[1] > 0.3 else 'LOW',
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        
        # Log prediction
        logger.info(f"Credit prediction: {prediction}, Risk score: {probability[1]:.3f}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in credit prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/explain/fraud', methods=['POST'])
def explain_fraud():
    """Get SHAP explanation for fraud prediction"""
    try:
        logger.info(f"Fraud explanation request from {request.remote_addr}")
        
        if fraud_model is None or fraud_explainer is None:
            return jsonify({'error': 'Fraud model or explainer not available'}), 500
        
        # Get and validate input
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        is_valid, message = validate_fraud_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Prepare features
        features = np.array([[data[field] for field in FRAUD_FEATURES]])
        features_scaled = fraud_scaler.transform(features)
        
        # Get SHAP values
        shap_values = fraud_explainer.shap_values(features_scaled)
        
        # Prepare explanation
        explanation = {
            'shap_values': {
                'non_fraud': shap_values[0][0].tolist(),
                'fraud': shap_values[1][0].tolist()
            },
            'feature_names': FRAUD_FEATURES,
            'feature_values': features[0].tolist(),
            'base_value': {
                'non_fraud': float(fraud_explainer.expected_value[0]),
                'fraud': float(fraud_explainer.expected_value[1])
            },
            'prediction': int(fraud_model.predict(features_scaled)[0]),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add top contributing features
        fraud_shap = shap_values[1][0]
        feature_contributions = list(zip(FRAUD_FEATURES, fraud_shap))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanation['top_contributors'] = [
            {'feature': feat, 'contribution': float(contrib)} 
            for feat, contrib in feature_contributions[:5]
        ]
        
        logger.info("Fraud explanation generated successfully")
        return jsonify(explanation), 200
        
    except Exception as e:
        logger.error(f"Error in fraud explanation: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/explain/credit', methods=['POST'])
def explain_credit():
    """Get SHAP explanation for credit card prediction"""
    try:
        logger.info(f"Credit explanation request from {request.remote_addr}")
        
        if credit_model is None or credit_explainer is None:
            return jsonify({'error': 'Credit model or explainer not available'}), 500
        
        # Get and validate input
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        is_valid, message = validate_credit_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Prepare features
        features = np.array([[data[field] for field in CREDIT_FEATURES]])
        features_scaled = credit_scaler.transform(features)
        
        # Get SHAP values
        shap_values = credit_explainer.shap_values(features_scaled)
        
        # Prepare explanation (top 10 features for readability)
        fraud_shap = shap_values[1][0]
        top_indices = np.argsort(np.abs(fraud_shap))[-10:]
        
        explanation = {
            'top_shap_values': fraud_shap[top_indices].tolist(),
            'top_feature_names': [CREDIT_FEATURES[i] for i in top_indices],
            'top_feature_values': features[0][top_indices].tolist(),
            'base_value': float(credit_explainer.expected_value[1]),
            'prediction': int(credit_model.predict(features_scaled)[0]),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add top contributing features
        feature_contributions = [(CREDIT_FEATURES[i], fraud_shap[i]) for i in top_indices]
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanation['top_contributors'] = [
            {'feature': feat, 'contribution': float(contrib)} 
            for feat, contrib in feature_contributions[:5]
        ]
        
        logger.info("Credit explanation generated successfully")
        return jsonify(explanation), 200
        
    except Exception as e:
        logger.error(f"Error in credit explanation: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch/fraud', methods=['POST'])
def batch_predict_fraud():
    """Batch prediction for multiple fraud transactions"""
    try:
        logger.info(f"Batch fraud prediction request from {request.remote_addr}")
        
        if fraud_model is None or fraud_scaler is None:
            return jsonify({'error': 'Fraud model not available'}), 500
        
        # Get and validate input
        data = request.get_json()
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions data provided'}), 400
        
        transactions = data['transactions']
        if not isinstance(transactions, list):
            return jsonify({'error': 'Transactions must be a list'}), 400
        
        results = []
        for i, transaction in enumerate(transactions):
            try:
                is_valid, message = validate_fraud_input(transaction)
                if not is_valid:
                    results.append({'index': i, 'error': message})
                    continue
                
                # Prepare features
                features = np.array([[transaction[field] for field in FRAUD_FEATURES]])
                features_scaled = fraud_scaler.transform(features)
                
                # Make prediction
                prediction = fraud_model.predict(features_scaled)[0]
                probability = fraud_model.predict_proba(features_scaled)[0]
                
                results.append({
                    'index': i,
                    'prediction': int(prediction),
                    'risk_score': float(probability[1]),
                    'risk_level': 'HIGH' if probability[1] > 0.7 else 'MEDIUM' if probability[1] > 0.3 else 'LOW'
                })
                
            except Exception as e:
                results.append({'index': i, 'error': str(e)})
        
        response = {
            'results': results,
            'total_processed': len(transactions),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch fraud prediction completed: {len(transactions)} transactions")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in batch fraud prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch/credit', methods=['POST'])
def batch_predict_credit():
    """Batch prediction for multiple credit card transactions"""
    try:
        logger.info(f"Batch credit prediction request from {request.remote_addr}")
        
        if credit_model is None or credit_scaler is None:
            return jsonify({'error': 'Credit model not available'}), 500
        
        # Get and validate input
        data = request.get_json()
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions data provided'}), 400
        
        transactions = data['transactions']
        if not isinstance(transactions, list):
            return jsonify({'error': 'Transactions must be a list'}), 400
        
        results = []
        for i, transaction in enumerate(transactions):
            try:
                is_valid, message = validate_credit_input(transaction)
                if not is_valid:
                    results.append({'index': i, 'error': message})
                    continue
                
                # Prepare features
                features = np.array([[transaction[field] for field in CREDIT_FEATURES]])
                features_scaled = credit_scaler.transform(features)
                
                # Make prediction
                prediction = credit_model.predict(features_scaled)[0]
                probability = credit_model.predict_proba(features_scaled)[0]
                
                results.append({
                    'index': i,
                    'prediction': int(prediction),
                    'risk_score': float(probability[1]),
                    'risk_level': 'HIGH' if probability[1] > 0.7 else 'MEDIUM' if probability[1] > 0.3 else 'LOW'
                })
                
            except Exception as e:
                results.append({'index': i, 'error': str(e)})
        
        response = {
            'results': results,
            'total_processed': len(transactions),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch credit prediction completed: {len(transactions)} transactions")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in batch credit prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Initialize models on import (for gunicorn)
logger.info("Loading models on module import...")
try:
    load_models()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    logger.error(traceback.format_exc())

if __name__ == '__main__':
    # Run the app directly (for development)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting development server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)