import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)

def load_dataset():
    """Load the health dataset from file"""
    try:
        # Default path for the dataset
        dataset_path = os.path.join('data', 'sample_health_data.csv')
        
        # Check if environment variable is set for alternate dataset location
        alternate_path = os.environ.get('HEALTH_DATASET_PATH')
        if alternate_path and os.path.exists(alternate_path):
            dataset_path = alternate_path
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded from {dataset_path}, shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        # Generate minimal synthetic data to avoid complete failure
        logger.warning("Using minimal synthetic dataset for demonstration")
        return create_minimal_dataset()

def create_minimal_dataset():
    """Create a minimal dataset for demonstration when the real one is not available"""
    # This is only used when the real dataset fails to load
    data = {
        'age': [25, 40, 35, 60, 55, 30, 45, 50, 65, 70],
        'gender': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 1 for male, 0 for female
        'heart_rate': [72, 75, 68, 80, 85, 70, 75, 82, 78, 76],
        'blood_pressure_systolic': [120, 130, 115, 145, 150, 125, 135, 140, 155, 160],
        'blood_pressure_diastolic': [80, 85, 75, 95, 100, 82, 88, 92, 98, 105],
        'cholesterol': [180, 200, 190, 250, 270, 185, 210, 230, 260, 280],
        'blood_sugar': [85, 90, 92, 110, 130, 88, 95, 100, 120, 140],
        'bmi': [22.5, 24.0, 23.5, 28.5, 30.0, 23.0, 25.5, 27.0, 29.5, 31.5],
        'smoking': [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
        'alcohol_consumption': [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
        'physical_activity': [5, 3, 4, 1, 0, 6, 2, 2, 1, 0],
        'family_history': [0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
        'risk_level': ['low', 'low', 'medium', 'high', 'high', 'low', 'medium', 'medium', 'high', 'high']
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the health dataset for training"""
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables
    if 'gender' in df.columns and df['gender'].dtype == 'object':
        df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    
    if 'risk_level' in df.columns:
        target = df['risk_level']
        features = df.drop('risk_level', axis=1)
    else:
        # If risk_level is not in the dataset, look for alternatives
        for col in ['health_risk', 'risk', 'outcome']:
            if col in df.columns:
                target = df[col]
                features = df.drop(col, axis=1)
                break
    
    # Standardize numeric features
    scaler = StandardScaler()
    numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
    features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
    
    return features, target, scaler

def train_model(df):
    """Train a Naive Bayes model on the dataset"""
    try:
        features, target, scaler = preprocess_data(df)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.25, random_state=42
        )
        
        # Train Naive Bayes model
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        logger.debug(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        return model, scaler
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        # Create a simple fallback model
        model = GaussianNB()
        scaler = StandardScaler()
        return model, scaler

def predict_health_risk(model, scaler, input_data):
    """Predict health risk based on input data"""
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply scaling
        numeric_cols = input_df.select_dtypes(include=['float64', 'int64']).columns
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        # Make prediction
        prediction_proba = model.predict_proba(input_df)[0]
        prediction_idx = np.argmax(prediction_proba)
        prediction = model.classes_[prediction_idx]
        probability = prediction_proba[prediction_idx]
        
        # Calculate risk score (0-100 scale)
        # For 'high' risk, we want a higher score
        if prediction.lower() == 'high':
            risk_score = 70 + (30 * probability)
        elif prediction.lower() == 'medium':
            risk_score = 30 + (40 * probability)
        else:  # low risk
            risk_score = 30 * probability
            
        return prediction, probability, risk_score
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Unknown", 0.0, 50.0

def get_feature_importance(model, scaler):
    """Extract feature importance from the trained model"""
    try:
        # Naive Bayes doesn't have direct feature importance, so we'll use variance
        # of the features which is stored in the variance_ attribute
        feature_names = scaler.feature_names_in_
        
        # In Gaussian Naive Bayes, we can use the variance (var_) for each class 
        # and feature as a rough measure of importance
        avg_var = np.mean(model.var_, axis=0)
        importance = avg_var / np.sum(avg_var)
        
        # Create and sort dict of feature importances
        importance_dict = dict(zip(feature_names, importance))
        sorted_importance = {k: float(v) for k, v in sorted(
            importance_dict.items(), key=lambda item: item[1], reverse=True
        )}
        
        return sorted_importance
    
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        return {}
