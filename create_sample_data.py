#!/usr/bin/env python3
"""
Create sample datasets for fraud detection challenge
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_fraud_data():
    """Create sample Fraud_Data.csv"""
    n_samples = 150000
    
    # Generate user IDs
    user_ids = np.random.randint(1, 500000, n_samples)
    
    # Generate signup times (2015 data)
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2015, 12, 31)
    signup_times = []
    for _ in range(n_samples):
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        signup_times.append(random_date.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Generate purchase times (after signup)
    purchase_times = []
    for signup in signup_times:
        signup_dt = datetime.strptime(signup, '%Y-%m-%d %H:%M:%S')
        # Purchase within 0-90 days after signup
        days_after = random.randint(0, 90)
        hours_after = random.randint(0, 23)
        minutes_after = random.randint(0, 59)
        purchase_dt = signup_dt + timedelta(days=days_after, hours=hours_after, minutes=minutes_after)
        purchase_times.append(purchase_dt.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Generate purchase values
    purchase_values = np.random.lognormal(3, 1, n_samples).astype(int)
    purchase_values = np.clip(purchase_values, 1, 1000)
    
    # Generate device IDs
    device_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    device_ids = [''.join(random.choices(device_chars, k=13)) for _ in range(n_samples)]
    
    # Generate sources
    sources = np.random.choice(['SEO', 'Ads', 'Direct'], n_samples, p=[0.4, 0.35, 0.25])
    
    # Generate browsers
    browsers = np.random.choice(['Chrome', 'Safari', 'Firefox', 'Opera', 'IE'], n_samples, 
                               p=[0.45, 0.25, 0.15, 0.1, 0.05])
    
    # Generate sex
    sex = np.random.choice(['M', 'F'], n_samples, p=[0.52, 0.48])
    
    # Generate ages
    ages = np.random.normal(35, 12, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    # Generate IP addresses (as integers)
    ip_addresses = np.random.uniform(1e8, 4e9, n_samples)
    
    # Generate fraud labels (imbalanced - 5% fraud)
    fraud_labels = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    # Create some patterns for fraud cases
    fraud_indices = np.where(fraud_labels == 1)[0]
    
    # Fraudulent transactions tend to:
    # - Have higher purchase values
    # - Happen quickly after signup
    # - Use certain browsers more often
    for idx in fraud_indices:
        if random.random() < 0.7:  # 70% of fraud cases have high values
            purchase_values[idx] = random.randint(200, 1000)
        if random.random() < 0.6:  # 60% happen within 1 day of signup
            signup_dt = datetime.strptime(signup_times[idx], '%Y-%m-%d %H:%M:%S')
            purchase_dt = signup_dt + timedelta(hours=random.randint(0, 24))
            purchase_times[idx] = purchase_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create DataFrame
    fraud_df = pd.DataFrame({
        'user_id': user_ids,
        'signup_time': signup_times,
        'purchase_time': purchase_times,
        'purchase_value': purchase_values,
        'device_id': device_ids,
        'source': sources,
        'browser': browsers,
        'sex': sex,
        'age': ages,
        'ip_address': ip_addresses,
        'class': fraud_labels
    })
    
    return fraud_df

def create_ip_country_data():
    """Create sample IpAddress_to_Country.csv"""
    countries = ['United States', 'China', 'India', 'Germany', 'United Kingdom', 
                'France', 'Brazil', 'Japan', 'Russia', 'Canada', 'Australia', 
                'South Korea', 'Italy', 'Spain', 'Mexico', 'Netherlands', 
                'Turkey', 'Saudi Arabia', 'Switzerland', 'Belgium']
    
    ip_ranges = []
    for i, country in enumerate(countries):
        # Create multiple IP ranges per country
        for j in range(random.randint(5, 15)):
            lower_bound = int(1e8 + i * 1e8 + j * 1e6)
            upper_bound = lower_bound + random.randint(1000, 100000)
            ip_ranges.append({
                'lower_bound_ip_address': lower_bound,
                'upper_bound_ip_address': upper_bound,
                'country': country
            })
    
    return pd.DataFrame(ip_ranges)

def create_creditcard_data():
    """Create sample creditcard.csv with PCA features"""
    n_samples = 284807  # Similar to original dataset size
    
    # Time feature (seconds elapsed)
    times = np.sort(np.random.uniform(0, 172792, n_samples))
    
    # Generate PCA features V1-V28
    pca_features = {}
    for i in range(1, 29):
        # Different distributions for different V features
        if i <= 10:
            pca_features[f'V{i}'] = np.random.normal(0, 1, n_samples)
        elif i <= 20:
            pca_features[f'V{i}'] = np.random.normal(0, 0.5, n_samples)
        else:
            pca_features[f'V{i}'] = np.random.normal(0, 2, n_samples)
    
    # Amount feature
    amounts = np.random.lognormal(3, 1.5, n_samples)
    amounts = np.clip(amounts, 0, 25691.16)
    
    # Class labels (highly imbalanced - 0.17% fraud)
    fraud_labels = np.random.choice([0, 1], n_samples, p=[0.9983, 0.0017])
    
    # Create some patterns for fraud
    fraud_indices = np.where(fraud_labels == 1)[0]
    for idx in fraud_indices:
        # Fraudulent transactions often have different V feature patterns
        for i in range(1, 15):
            if random.random() < 0.3:
                pca_features[f'V{i}'][idx] *= random.uniform(2, 5)
    
    # Create DataFrame
    credit_df = pd.DataFrame({
        'Time': times,
        **pca_features,
        'Amount': amounts,
        'Class': fraud_labels
    })
    
    return credit_df

def main():
    print("Creating sample datasets...")
    
    # Create Fraud_Data.csv
    print("Creating Fraud_Data.csv...")
    fraud_data = create_fraud_data()
    fraud_data.to_csv('/workspace/data/Fraud_Data.csv', index=False)
    print(f"Created Fraud_Data.csv with {len(fraud_data)} records")
    
    # Create IpAddress_to_Country.csv
    print("Creating IpAddress_to_Country.csv...")
    ip_data = create_ip_country_data()
    ip_data.to_csv('/workspace/data/IpAddress_to_Country.csv', index=False)
    print(f"Created IpAddress_to_Country.csv with {len(ip_data)} records")
    
    # Create creditcard.csv
    print("Creating creditcard.csv...")
    credit_data = create_creditcard_data()
    credit_data.to_csv('/workspace/data/creditcard.csv', index=False)
    print(f"Created creditcard.csv with {len(credit_data)} records")
    
    print("All datasets created successfully!")
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Fraud_Data.csv: {len(fraud_data)} records, {fraud_data['class'].sum()} fraud cases ({fraud_data['class'].mean()*100:.2f}%)")
    print(f"IpAddress_to_Country.csv: {len(ip_data)} IP ranges")
    print(f"creditcard.csv: {len(credit_data)} records, {credit_data['Class'].sum()} fraud cases ({credit_data['Class'].mean()*100:.2f}%)")

if __name__ == "__main__":
    main()