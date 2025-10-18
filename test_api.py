#!/usr/bin/env python3
"""
API Testing Script for Advanced Fraud Detection System
"""

import requests
import json
import time
import sys
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:5000"
TIMEOUT = 30

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check Passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Models: {data.get('models')}")
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

def test_fraud_prediction():
    """Test fraud prediction endpoint"""
    print("\n🔍 Testing Fraud Prediction...")
    
    # Test data
    test_cases = [
        {
            "name": "Low Risk Transaction",
            "data": {
                "purchase_value": 50.0,
                "age": 35,
                "hour_of_day": 14,
                "day_of_week": 2,
                "source_encoded": 0,
                "browser_encoded": 0,
                "sex_encoded": 1
            }
        },
        {
            "name": "High Risk Transaction",
            "data": {
                "purchase_value": 500.0,
                "age": 25,
                "hour_of_day": 3,
                "day_of_week": 6,
                "source_encoded": 2,
                "browser_encoded": 1,
                "sex_encoded": 0
            }
        }
    ]
    
    success_count = 0
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict/fraud",
                json=test_case["data"],
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {test_case['name']}")
                print(f"   Prediction: {'FRAUD' if data['prediction'] == 1 else 'NOT FRAUD'}")
                print(f"   Risk Score: {data['risk_score']:.3f}")
                print(f"   Risk Level: {data['risk_level']}")
                success_count += 1
            else:
                print(f"❌ {test_case['name']} Failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ {test_case['name']} Error: {e}")
    
    return success_count == len(test_cases)

def test_credit_prediction():
    """Test credit card prediction endpoint"""
    print("\n🔍 Testing Credit Card Prediction...")
    
    # Generate test data for credit card (30 features)
    test_data = {
        "Time": 12345.0,
        "Amount": 100.0
    }
    
    # Add V1-V28 features with random-like values
    for i in range(1, 29):
        test_data[f"V{i}"] = 0.1 * i  # Simple test values
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/credit",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Credit Card Prediction")
            print(f"   Prediction: {'FRAUD' if data['prediction'] == 1 else 'NOT FRAUD'}")
            print(f"   Risk Score: {data['risk_score']:.3f}")
            print(f"   Risk Level: {data['risk_level']}")
            return True
        else:
            print(f"❌ Credit Card Prediction Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Credit Card Prediction Error: {e}")
        return False

def test_fraud_explanation():
    """Test fraud explanation endpoint"""
    print("\n🔍 Testing Fraud Explanation...")
    
    test_data = {
        "purchase_value": 150.0,
        "age": 35,
        "hour_of_day": 14,
        "day_of_week": 2,
        "source_encoded": 1,
        "browser_encoded": 0,
        "sex_encoded": 1
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/explain/fraud",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Fraud Explanation")
            print(f"   Prediction: {'FRAUD' if data['prediction'] == 1 else 'NOT FRAUD'}")
            print(f"   Top Contributors:")
            for contrib in data.get('top_contributors', [])[:3]:
                print(f"     {contrib['feature']}: {contrib['contribution']:.3f}")
            return True
        else:
            print(f"❌ Fraud Explanation Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Fraud Explanation Error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n🔍 Testing Batch Prediction...")
    
    batch_data = {
        "transactions": [
            {
                "purchase_value": 50.0,
                "age": 35,
                "hour_of_day": 14,
                "day_of_week": 2,
                "source_encoded": 0,
                "browser_encoded": 0,
                "sex_encoded": 1
            },
            {
                "purchase_value": 300.0,
                "age": 28,
                "hour_of_day": 2,
                "day_of_week": 6,
                "source_encoded": 1,
                "browser_encoded": 1,
                "sex_encoded": 0
            },
            {
                "purchase_value": 75.0,
                "age": 45,
                "hour_of_day": 10,
                "day_of_week": 3,
                "source_encoded": 0,
                "browser_encoded": 0,
                "sex_encoded": 1
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch/fraud",
            json=batch_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch Prediction")
            print(f"   Total Processed: {data['total_processed']}")
            print(f"   Successful: {data['successful_predictions']}")
            
            for i, result in enumerate(data['results'][:3]):
                if 'error' not in result:
                    print(f"   Transaction {i}: {'FRAUD' if result['prediction'] == 1 else 'NOT FRAUD'} "
                          f"(Risk: {result['risk_score']:.3f})")
                else:
                    print(f"   Transaction {i}: Error - {result['error']}")
            return True
        else:
            print(f"❌ Batch Prediction Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch Prediction Error: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n🔍 Testing Error Handling...")
    
    # Test invalid endpoint
    try:
        response = requests.get(f"{API_BASE_URL}/invalid-endpoint", timeout=TIMEOUT)
        if response.status_code == 404:
            print("✅ 404 Error Handling")
        else:
            print(f"❌ Expected 404, got {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing invalid endpoint: {e}")
    
    # Test invalid JSON
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/fraud",
            json={"invalid": "data"},
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        if response.status_code == 400:
            print("✅ Invalid Input Error Handling")
        else:
            print(f"❌ Expected 400, got {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing invalid JSON: {e}")

def performance_test():
    """Basic performance test"""
    print("\n🔍 Performance Test...")
    
    test_data = {
        "purchase_value": 100.0,
        "age": 35,
        "hour_of_day": 14,
        "day_of_week": 2,
        "source_encoded": 1,
        "browser_encoded": 0,
        "sex_encoded": 1
    }
    
    num_requests = 10
    start_time = time.time()
    success_count = 0
    
    for i in range(num_requests):
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict/fraud",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                success_count += 1
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_requests
    
    print(f"✅ Performance Test Results:")
    print(f"   Total Requests: {num_requests}")
    print(f"   Successful: {success_count}")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Average Response Time: {avg_time:.3f}s")
    print(f"   Requests per Second: {num_requests/total_time:.2f}")

def main():
    """Run all tests"""
    print("🚀 Starting API Tests for Advanced Fraud Detection System")
    print(f"📅 Test Run: {datetime.now().isoformat()}")
    print(f"🌐 API URL: {API_BASE_URL}")
    print("=" * 60)
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready!")
                break
        except:
            pass
        
        if i == max_retries - 1:
            print("❌ API not responding. Please check if the server is running.")
            sys.exit(1)
        
        time.sleep(2)
        print(f"   Retry {i+1}/{max_retries}...")
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Fraud Prediction", test_fraud_prediction),
        ("Credit Prediction", test_credit_prediction),
        ("Fraud Explanation", test_fraud_explanation),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling),
        ("Performance Test", performance_test)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! API is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please check the API implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()