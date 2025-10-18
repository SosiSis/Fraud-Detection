#!/usr/bin/env python3
"""
Train simple models for API deployment
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def train_fraud_model():
    """Train fraud detection model"""
    print("Training fraud detection model...")
    
    # Load data
    data = pd.read_csv('data/Fraud_Data.csv')
    
    # Basic preprocessing
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    data['hour_of_day'] = data['purchase_time'].dt.hour
    data['day_of_week'] = data['purchase_time'].dt.dayofweek
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in ['source', 'browser', 'sex']:
        data[f'{col}_encoded'] = le.fit_transform(data[col])
    
    # Select features
    features = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 
               'source_encoded', 'browser_encoded', 'sex_encoded']
    
    X = data[features]
    y = data['class']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Fraud model - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_fraud_model.pkl')
    joblib.dump(scaler, 'models/fraud_scaler.pkl')
    
    print("Fraud model saved successfully!")
    return model, scaler

def train_credit_model():
    """Train credit card fraud detection model"""
    print("Training credit card fraud detection model...")
    
    # Load data
    data = pd.read_csv('data/creditcard.csv')
    
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Credit model - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_credit_model.pkl')
    joblib.dump(scaler, 'models/credit_scaler.pkl')
    
    print("Credit model saved successfully!")
    return model, scaler

def main():
    print("🤖 Training models for API deployment...")
    print("=" * 50)
    
    try:
        # Train models
        fraud_model, fraud_scaler = train_fraud_model()
        credit_model, credit_scaler = train_credit_model()
        
        print("\n✅ All models trained and saved successfully!")
        print("Models are ready for API deployment.")
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()