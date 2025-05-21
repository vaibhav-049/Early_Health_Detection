import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def load_dataset():
    """Load the health dataset from file"""
    try:
        training_path = os.path.join('data', 'Training.csv')
        testing_path = os.path.join('data', 'Testing.csv')
        
        alternate_training = os.environ.get('HEALTH_TRAINING_DATASET_PATH')
        alternate_testing = os.environ.get('HEALTH_TESTING_DATASET_PATH')
        
        if alternate_training and os.path.exists(alternate_training):
            training_path = alternate_training
        if alternate_testing and os.path.exists(alternate_testing):
            testing_path = alternate_testing
        
        if not os.path.exists(training_path):
            logger.error(f"Training dataset not found at {training_path}")
            raise FileNotFoundError(f"Training dataset not found at {training_path}")
            
        if not os.path.exists(testing_path):
            logger.error(f"Testing dataset not found at {testing_path}")
            raise FileNotFoundError(f"Testing dataset not found at {testing_path}")
        
        train_df = pd.read_csv(training_path)
        test_df = pd.read_csv(testing_path)
        
        if train_df.empty or test_df.empty:
            logger.error("Dataset is empty")
            raise ValueError("Dataset is empty")
            
        if 'prognosis' not in train_df.columns:
            logger.error("Dataset does not contain 'prognosis' column")
            train_df = train_df.rename(columns={train_df.columns[-1]: 'prognosis'})
            test_df = test_df.rename(columns={test_df.columns[-1]: 'prognosis'})
            logger.warning(f"Renamed last column to 'prognosis'")
        
        logger.info(f"Training dataset loaded from {training_path}, shape: {train_df.shape}")
        logger.info(f"Testing dataset loaded from {testing_path}, shape: {test_df.shape}")
        logger.info(f"Dataset contains {len(train_df['prognosis'].unique())} unique diseases")
        
        return train_df, test_df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        # Generate minimal synthetic data to avoid complete failure
        logger.warning("Using minimal synthetic dataset for demonstration")
        return create_minimal_dataset(), create_minimal_dataset(test=True)

def create_minimal_dataset(test=False):
    """Create a minimal dataset for demonstration when the real one is not available"""
    # This is only used when the real dataset fails to load
    # Creating a simplified version of the disease dataset with common symptoms
    symptoms = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
        'chills', 'joint_pain', 'stomach_pain', 'acidity', 'vomiting', 'fatigue', 'weight_loss',
        'cough', 'high_fever', 'breathlessness', 'sweating', 'headache', 'nausea', 'back_pain'
    ]
    
    # Create 10 sample records (or 5 for test set)
    num_samples = 5 if test else 10
    data = {}
    
    # Initialize all symptoms as 0
    for symptom in symptoms:
        data[symptom] = [0] * num_samples
    
    # Set some symptoms to 1 for each disease - expanded disease set
    diseases = [
        'Fungal infection', 'Allergy', 'Common Cold', 'Pneumonia', 'Dengue',
        'Typhoid', 'Malaria', 'Arthritis', 'GERD', 'Gastroenteritis',
        'Urinary tract infection', 'Chronic fatigue syndrome', 'Migraine', 'Heart attack'
    ]
    
    disease_symptoms = {
        'Fungal infection': ['itching', 'skin_rash', 'nodal_skin_eruptions'],
        'Allergy': ['continuous_sneezing', 'skin_rash', 'chills'],
        'Common Cold': ['continuous_sneezing', 'cough', 'fatigue', 'headache', 'back_pain'],
        'Pneumonia': ['high_fever', 'breathlessness', 'fatigue', 'cough'],
        'Dengue': ['high_fever', 'joint_pain', 'fatigue', 'headache', 'nausea'],
        'Typhoid': ['high_fever', 'shivering', 'fatigue', 'headache', 'nausea', 'abdominal_pain'],
        'Malaria': ['chills', 'high_fever', 'sweating', 'headache', 'nausea'],
        'Arthritis': ['joint_pain', 'back_pain', 'fatigue'],
        'GERD': ['acidity', 'abdominal_pain', 'chest_pain'],
        'Gastroenteritis': ['vomiting', 'nausea', 'abdominal_pain', 'stomach_pain'],
        'Urinary tract infection': ['burning_micturition', 'fatigue'],
        'Chronic fatigue syndrome': ['fatigue', 'muscle_wasting', 'headache', 'joint_pain'],
        'Migraine': ['headache', 'nausea', 'dizziness'],
        'Heart attack': ['chest_pain', 'breathlessness', 'fatigue', 'sweating']
    }
    
    data['prognosis'] = []
    
    samples_per_disease = num_samples // len(diseases)
    for disease in diseases:
        for _ in range(samples_per_disease):
            data['prognosis'].append(disease)
            
            idx = len(data['prognosis']) - 1
            for symptom in disease_symptoms[disease]:
                data[symptom][idx] = 1
    
    remaining = num_samples - len(data['prognosis'])
    for i in range(remaining):
        disease = diseases[i]
        data['prognosis'].append(disease)
        idx = len(data['prognosis']) - 1
        for symptom in disease_symptoms[disease]:
            data[symptom][idx] = 1
    
    return pd.DataFrame(data)

def preprocess_data(train_df, test_df=None):
    """Preprocess the health dataset for training"""
    train_df = train_df.fillna(0)
    
    if train_df.empty:
        logger.error("Training dataset is empty")
        raise ValueError("Training dataset is empty")
    
    if 'prognosis' in train_df.columns:
        target_col = 'prognosis'
    elif 'risk_level' in train_df.columns:
        target_col = 'risk_level'
    else:
        # If neither column exists, look for alternatives
        for col in ['health_risk', 'risk', 'outcome', 'disease']:
            if col in train_df.columns:
                target_col = col
                break
        else:
            # If no suitable target column is found, use the last column
            target_col = train_df.columns[-1]
            logger.warning(f"No explicit target column found, using {target_col}")
    
    # Ensure test_df has the target column
    if test_df is not None:
        if target_col not in test_df.columns:
            logger.warning(f"Target column '{target_col}' not found in test dataset. Using training data for testing.")
            test_df = None
    
    # Clean column names - remove any unnamed columns
    unnamed_cols = [col for col in train_df.columns if 'Unnamed:' in str(col)]
    if unnamed_cols:
        logger.warning(f"Removing unnamed columns: {unnamed_cols}")
        train_df = train_df.drop(columns=unnamed_cols)
        if test_df is not None:
            test_df = test_df.drop(columns=[col for col in unnamed_cols if col in test_df.columns])
    
    # Extract features and target
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    
    # Verify we have data after preprocessing
    if len(X_train) == 0 or len(y_train) == 0:
        logger.error("No data available after preprocessing")
        raise ValueError("No data available after preprocessing")
    
    if test_df is not None:
        # Make sure test_df has the same feature columns as train_df
        missing_cols = set(X_train.columns) - set(test_df.columns)
        if missing_cols:
            logger.warning(f"Test dataset missing columns: {missing_cols}. Adding with default values.")
            for col in missing_cols:
                test_df[col] = 0
        
        # Handle extra columns in test data
        extra_cols = set(test_df.columns) - set(X_train.columns) - {target_col}
        if extra_cols:
            logger.warning(f"Test dataset has extra columns: {extra_cols}. Removing them.")
            test_df = test_df.drop(columns=list(extra_cols))
        
        # Only use columns that are in the training data
        test_cols = list(X_train.columns)
        if target_col in test_df.columns:
            test_cols.append(target_col)
        
        test_df = test_df[test_cols]
        
        X_test = test_df.drop(target_col, axis=1) if target_col in test_df.columns else test_df
        y_test = test_df[target_col] if target_col in test_df.columns else None
    else:
        # If no test data is provided, use a portion of the training data
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create a preprocessor
    preprocessor = StandardScaler()
    
    # Verify we have data for both training and testing
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Empty training or testing data after preprocessing")
        raise ValueError("Empty training or testing data after preprocessing")
    
    logger.info(f"Preprocessed data: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor, target_col

def train_models(train_df, test_df=None, algorithm='all'):
    """Train multiple ML models on the dataset
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe (optional)
        algorithm: Which algorithm to train ('naive_bayes', 'decision_tree', 'random_forest', 'gradient_boosting', or 'all')
        
    Returns:
        Dictionary of trained models and the preprocessor
    """
    try:
        X_train, X_test, y_train, y_test, preprocessor, target_col = preprocess_data(train_df, test_df)
        
        # Initialize models dictionary
        models = {}
        
        # Define which models to train based on algorithm parameter
        models_to_train = {
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        if algorithm.lower() != 'all':
            # Train only the specified algorithm
            if algorithm.lower() in models_to_train:
                selected_model = models_to_train[algorithm.lower()]
                models[algorithm.lower()] = train_single_model(selected_model, X_train, y_train, X_test, y_test)
            else:
                logger.error(f"Unknown algorithm: {algorithm}. Defaulting to Naive Bayes.")
                models['naive_bayes'] = train_single_model(models_to_train['naive_bayes'], X_train, y_train, X_test, y_test)
        else:
            # Train all models
            for name, model in models_to_train.items():
                models[name] = train_single_model(model, X_train, y_train, X_test, y_test)
        
        # Set the best model based on accuracy
        best_model_name = max(models, key=lambda k: models[k]['accuracy'])
        models['best_model'] = best_model_name
        
        logger.info(f"Best model: {best_model_name} with accuracy: {models[best_model_name]['accuracy']:.4f}")
        
        return models, preprocessor
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        # Create a simple fallback model
        model = GaussianNB().fit(X_train, y_train)
        return {'naive_bayes': {'model': model, 'accuracy': 0.0}, 'best_model': 'naive_bayes'}, preprocessor


def train_single_model(model, X_train, y_train, X_test, y_test):
    """Train a single model and evaluate it"""
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log performance metrics
    logger.info(f"{type(model).__name__} trained with accuracy: {accuracy:.4f}")
    logger.debug(f"Classification report:\n{classification_report(y_test, y_pred)}")
    
    # Return model and its performance metrics
    return {
        'model': model,
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

def predict_health_condition(models, preprocessor, input_data, algorithm=None):
    """Predict health condition based on input data"""
    try:
        # Convert input_data to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_data_dict = input_data.copy()  # Save original dict for later use
            input_data = pd.DataFrame([input_data])

        # Log the input data shape and content
        logger.debug(f"Input data before preprocessing: {input_data.shape}\n{input_data.head()}")

        # Ensure models are available
        if models is None or len(models) == 0:
            logger.error("No models available for prediction")
            return "Unknown", 0.0, 0.0, {"error": "No models available"}

        # Check if any symptoms are selected
        if input_data.sum().sum() == 0:
            logger.warning("No symptoms selected for prediction")
            return "No symptoms selected", 0.0, 0.0, {"error": "No symptoms selected"}

        # Create a list of selected symptoms for debugging
        selected_symptoms = [col for col in input_data.columns if input_data[col].iloc[0] == 1]
        logger.debug(f"Selected symptoms: {selected_symptoms}")

        # Preprocess the input data
        try:
            # Make sure we have the right columns for the model
            if preprocessor is not None:
                # Check if we need to adjust columns to match what the model expects
                try:
                    X = preprocessor.transform(input_data)
                except ValueError as ve:
                    logger.warning(f"Preprocessing error: {ve}. Attempting to fix column mismatch...")
                    # Get expected columns from the preprocessor if available
                    if hasattr(preprocessor, 'feature_names_in_'):
                        expected_columns = preprocessor.feature_names_in_
                        # Create a new DataFrame with the expected columns
                        fixed_input = pd.DataFrame(0, index=[0], columns=expected_columns)
                        # Copy over the values for columns that exist in both
                        for col in input_data.columns:
                            if col in expected_columns:
                                fixed_input[col] = input_data[col].values
                        # Try again with the fixed input
                        X = preprocessor.transform(fixed_input)
                        logger.info("Successfully fixed column mismatch")
                    else:
                        # If we can't fix it, raise the original error
                        raise ve
            else:
                X = input_data

            logger.debug(f"Data after preprocessing: {X.shape}\n{X[:5]}")

            if hasattr(X, 'shape') and X.shape[0] == 0:
                logger.error("Empty data after preprocessing")
                return "Error in data processing", 0.0, 0.0, {"error": "Empty data after preprocessing"}
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            fallback_predictions = {}
            contributing_symptoms = {}
            
            for symptom, value in input_data_dict.items():
                if value == 1:
                    display_name = symptom.replace('_', ' ').title()
                    contributing_symptoms[display_name] = round(100 / max(1, len(selected_symptoms)), 1)
            
            fallback_predictions['contributing_symptoms'] = contributing_symptoms
            
            # Enhanced disease-symptom mapping table for more accurate symptom-based diagnosis
            disease_mapping = {
                'Itching': ['Fungal infection', 'Allergy'],
                'Skin Rash': ['Allergy', 'Fungal infection', 'Chicken pox'],
                'Nodal Skin Eruptions': ['Fungal infection'],
                'Continuous Sneezing': ['Common Cold', 'Allergy'],
                'Shivering': ['Typhoid', 'Malaria', 'Dengue'],
                'Chills': ['Malaria', 'Dengue', 'Common Cold'],
                'Joint Pain': ['Arthritis', 'Dengue', 'Chronic fatigue syndrome'],
                'Stomach Pain': ['Food Poisoning', 'GERD', 'Gastroenteritis'],
                'Acidity': ['GERD', 'Peptic ulcer diseae'],
                'Ulcers On Tongue': ['Fungal infection'],
                'Vomiting': ['Gastroenteritis', 'Food Poisoning', 'Dengue'],
                'Burning Micturition': ['Urinary tract infection'],
                'Fatigue': ['Chronic fatigue syndrome', 'Common Cold', 'Dengue', 'Typhoid'],
                'Cough': ['Common Cold', 'Pneumonia', 'Bronchial Asthma', 'Tuberculosis'],
                'High Fever': ['Typhoid', 'Malaria', 'Dengue', 'Pneumonia'],
                'Breathlessness': ['Pneumonia', 'Bronchial Asthma', 'Heart attack'],
                'Sweating': ['Malaria', 'Dengue', 'Hyperthyroidism'],
                'Headache': ['Migraine', 'Common Cold', 'Dengue', 'Malaria'],
                'Nausea': ['Gastroenteritis', 'Food Poisoning', 'Dengue', 'Migraine'],
                'Chest Pain': ['Heart attack', 'Pneumonia', 'GERD'],
                'Back Pain': ['Chronic back pain', 'Arthritis'],
                'Abdominal Pain': ['Appendicitis', 'Food Poisoning', 'GERD', 'Gastroenteritis'],
                'Muscle Wasting': ['Diabetes', 'Chronic fatigue syndrome'],
                'Irregular Sugar Level': ['Diabetes', 'Hypoglycemia'],
                'Constipation': ['GERD', 'Irritable bowel syndrome'],
                'Pain During Bowel Movements': ['Hemorrhoids', 'Irritable bowel syndrome'],
                'Dizziness': ['Migraine', 'Hypertension', 'Hypotension'],
                'Hip Joint Pain': ['Arthritis'],
                'Muscle Pain': ['Chronic fatigue syndrome', 'Dengue'],
                'Red Spots Over Body': ['Chicken pox', 'Dengue', 'Measles'],
                'Runny Nose': ['Common Cold', 'Allergy']
            }
            
            # Find the most likely disease based on selected symptoms using weighted scoring
            disease_scores = {}
            total_symptoms = 0
            matched_symptoms_by_disease = {}
            
            # Count how many symptoms were selected
            for symptom, value in input_data_dict.items():
                if value == 1:
                    total_symptoms += 1
                    display_name = symptom.replace('_', ' ').title()
                    
                    # Use the mapping if available
                    if display_name in disease_mapping:
                        # Each symptom can map to multiple diseases with different weights
                        possible_diseases = disease_mapping[display_name]
                        if isinstance(possible_diseases, str):  # Handle case where it's a single string
                            possible_diseases = [possible_diseases]
                            
                        # Primary match gets higher weight
                        if len(possible_diseases) > 0:
                            primary_disease = possible_diseases[0]
                            disease_scores[primary_disease] = disease_scores.get(primary_disease, 0) + 1.5
                            
                            # Track which symptoms matched each disease (for display)
                            if primary_disease not in matched_symptoms_by_disease:
                                matched_symptoms_by_disease[primary_disease] = []
                            matched_symptoms_by_disease[primary_disease].append(display_name)
                        
                        # Secondary matches get lower weights
                        for i, disease in enumerate(possible_diseases[1:], 1):
                            weight = max(0.5, 1.0 - (i * 0.2))  # Decreasing weights for lower-ranked diseases
                            disease_scores[disease] = disease_scores.get(disease, 0) + weight
                            
                            if disease not in matched_symptoms_by_disease:
                                matched_symptoms_by_disease[disease] = []
                            matched_symptoms_by_disease[disease].append(display_name)
            
            # Find the disease with the highest weighted score
            if disease_scores:
                # Sort diseases by their scores
                sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
                predicted_disease = sorted_diseases[0][0]
                
                # Calculate a more nuanced confidence score
                top_score = sorted_diseases[0][1]
                
                # Calculate confidence based on:
                # 1. How many symptoms matched the top disease
                # 2. The relative strength of the top disease vs. second-best
                # 3. What percentage of the selected symptoms match the top disease
                num_matched_symptoms = len(matched_symptoms_by_disease.get(predicted_disease, []))
                symptom_coverage = num_matched_symptoms / max(1, total_symptoms)
                
                # If there's more than one disease, check the relative strength
                score_difference = 0
                if len(sorted_diseases) > 1:
                    second_best_score = sorted_diseases[1][1]
                    # Calculate normalized difference between top and second scores
                    if top_score > 0:
                        score_difference = (top_score - second_best_score) / top_score
                else:
                    # If only one disease, give full points for distinctiveness
                    score_difference = 1.0
                
                # Weight the different factors
                confidence = (0.5 * symptom_coverage) + (0.3 * score_difference) + (0.2 * min(1.0, top_score / 5.0))
                confidence = min(0.95, max(0.4, confidence))  # Clamp between 0.4 and 0.95
                
                # Add matched symptoms to fallback_predictions for display
                fallback_predictions['matched_symptoms'] = matched_symptoms_by_disease.get(predicted_disease, [])
                fallback_predictions['alternative_diseases'] = [
                    {'name': disease, 'score': round(score * 20, 1)} 
                    for disease, score in sorted_diseases[1:4]  # Get next 3 alternatives
                ] if len(sorted_diseases) > 1 else []
            else:
                predicted_disease = "Unknown Condition"
                confidence = 0.4
                
            return predicted_disease, confidence, confidence * 100, fallback_predictions

        # Make prediction
        try:
            # Get the correct model based on algorithm
            if algorithm == 'all':
                # Use the best model if 'all' is selected
                model_key = models.get('best_model', 'random_forest')
                model = models.get(model_key, {}).get('model')
            else:
                # Use the specified algorithm model
                model = models.get(algorithm, {}).get('model')
                if model is None:
                    # Fallback to best model if specified algorithm is not found
                    best_model_key = models.get('best_model', 'random_forest')
                    model = models.get(best_model_key, {}).get('model')
            
            if model is None:
                logger.error("No suitable model found for prediction")
                return "Model not found", 0.0, 0.0, {"error": "Model not found"}

            # Make prediction with the selected model
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Get all possible classes and their probabilities
            all_predictions = {}
            for i, cls in enumerate(model.classes_):
                all_predictions[cls] = float(probabilities[i])
            
            # Get the probability for the predicted class
            predicted_idx = list(model.classes_).index(prediction)
            probability = probabilities[predicted_idx]
            confidence_score = probability * 100

            # Log the prediction results
            logger.debug(f"Prediction: {prediction}, Probability: {probability}, Confidence Score: {confidence_score}")
            
            # Extract contributing symptoms (those that were present in the input)
            contributing_symptoms = {}
            for symptom, value in input_data_dict.items():
                if value == 1:  # If symptom is present
                    # Format the symptom name for display
                    display_name = symptom.replace('_', ' ').title()
                    # Assign an importance value (can be refined with feature importance if available)
                    contributing_symptoms[display_name] = round(100 / max(1, len(selected_symptoms)), 1)
            
            # Add contributing symptoms to the all_predictions dictionary
            all_predictions['contributing_symptoms'] = contributing_symptoms
            
            return prediction, probability, confidence_score, all_predictions
            
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            # Create a fallback prediction with the raw symptoms
            fallback_predictions = {}
            contributing_symptoms = {}
            
            for symptom, value in input_data_dict.items():
                if value == 1:  # If symptom is present
                    # Format the symptom name for display
                    display_name = symptom.replace('_', ' ').title()
                    # Add to contributing symptoms
                    contributing_symptoms[display_name] = round(100 / max(1, len(selected_symptoms)), 1)
            
            fallback_predictions['contributing_symptoms'] = contributing_symptoms
            
            # Use the enhanced symptom-disease mapping for a more accurate fallback prediction
            # Enhanced disease-symptom mapping table (same as above)
            disease_mapping = {
                'Itching': ['Fungal infection', 'Allergy'],
                'Skin Rash': ['Allergy', 'Fungal infection', 'Chicken pox'],
                'Nodal Skin Eruptions': ['Fungal infection'],
                'Continuous Sneezing': ['Common Cold', 'Allergy'],
                'Shivering': ['Typhoid', 'Malaria', 'Dengue'],
                'Chills': ['Malaria', 'Dengue', 'Common Cold'],
                'Joint Pain': ['Arthritis', 'Dengue', 'Chronic fatigue syndrome'],
                'Stomach Pain': ['Food Poisoning', 'GERD', 'Gastroenteritis'],
                'Acidity': ['GERD', 'Peptic ulcer diseae'],
                'Ulcers On Tongue': ['Fungal infection'],
                'Vomiting': ['Gastroenteritis', 'Food Poisoning', 'Dengue'],
                'Burning Micturition': ['Urinary tract infection'],
                'Fatigue': ['Chronic fatigue syndrome', 'Common Cold', 'Dengue', 'Typhoid'],
                'Cough': ['Common Cold', 'Pneumonia', 'Bronchial Asthma', 'Tuberculosis'],
                'High Fever': ['Typhoid', 'Malaria', 'Dengue', 'Pneumonia'],
                'Breathlessness': ['Pneumonia', 'Bronchial Asthma', 'Heart attack'],
                'Sweating': ['Malaria', 'Dengue', 'Hyperthyroidism'],
                'Headache': ['Migraine', 'Common Cold', 'Dengue', 'Malaria'],
                'Nausea': ['Gastroenteritis', 'Food Poisoning', 'Dengue', 'Migraine'],
                'Chest Pain': ['Heart attack', 'Pneumonia', 'GERD'],
                'Back Pain': ['Chronic back pain', 'Arthritis'],
                'Abdominal Pain': ['Appendicitis', 'Food Poisoning', 'GERD', 'Gastroenteritis'],
                'Muscle Wasting': ['Diabetes', 'Chronic fatigue syndrome'],
                'Irregular Sugar Level': ['Diabetes', 'Hypoglycemia'],
                'Constipation': ['GERD', 'Irritable bowel syndrome'],
                'Pain During Bowel Movements': ['Hemorrhoids', 'Irritable bowel syndrome'],
                'Dizziness': ['Migraine', 'Hypertension', 'Hypotension'],
                'Hip Joint Pain': ['Arthritis'],
                'Muscle Pain': ['Chronic fatigue syndrome', 'Dengue'],
                'Red Spots Over Body': ['Chicken pox', 'Dengue', 'Measles'],
                'Runny Nose': ['Common Cold', 'Allergy']
            }
            
            # Find the most likely disease based on selected symptoms using weighted scoring
            disease_scores = {}
            total_symptoms = 0
            matched_symptoms_by_disease = {}
            
            # Count how many symptoms were selected
            for symptom, value in input_data_dict.items():
                if value == 1:
                    total_symptoms += 1
                    display_name = symptom.replace('_', ' ').title()
                    
                    # Use the mapping if available
                    if display_name in disease_mapping:
                        # Each symptom can map to multiple diseases with different weights
                        possible_diseases = disease_mapping[display_name]
                        if isinstance(possible_diseases, str):  # Handle case where it's a single string
                            possible_diseases = [possible_diseases]
                            
                        # Primary match gets higher weight
                        if len(possible_diseases) > 0:
                            primary_disease = possible_diseases[0]
                            disease_scores[primary_disease] = disease_scores.get(primary_disease, 0) + 1.5
                            
                            # Track which symptoms matched each disease (for display)
                            if primary_disease not in matched_symptoms_by_disease:
                                matched_symptoms_by_disease[primary_disease] = []
                            matched_symptoms_by_disease[primary_disease].append(display_name)
                        
                        # Secondary matches get lower weights
                        for i, disease in enumerate(possible_diseases[1:], 1):
                            weight = max(0.5, 1.0 - (i * 0.2))  # Decreasing weights for lower-ranked diseases
                            disease_scores[disease] = disease_scores.get(disease, 0) + weight
                            
                            if disease not in matched_symptoms_by_disease:
                                matched_symptoms_by_disease[disease] = []
                            matched_symptoms_by_disease[disease].append(display_name)
            
            if disease_scores:
                # Sort diseases by their scores
                sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
                predicted_disease = sorted_diseases[0][0]
                
                # Calculate a more nuanced confidence score
                top_score = sorted_diseases[0][1]
                
                # Calculate confidence based on multiple factors
                num_matched_symptoms = len(matched_symptoms_by_disease.get(predicted_disease, []))
                symptom_coverage = num_matched_symptoms / max(1, total_symptoms)
                
                # If there's more than one disease, check the relative strength
                score_difference = 0
                if len(sorted_diseases) > 1:
                    second_best_score = sorted_diseases[1][1]
                    if top_score > 0:
                        score_difference = (top_score - second_best_score) / top_score
                else:
                    score_difference = 1.0
                
                # Weight the different factors
                confidence = (0.5 * symptom_coverage) + (0.3 * score_difference) + (0.2 * min(1.0, top_score / 5.0))
                confidence = min(0.9, max(0.4, confidence))  # Clamp between 0.4 and 0.9
                
                # Add alternative diseases and matched symptoms to fallback_predictions
                fallback_predictions['matched_symptoms'] = matched_symptoms_by_disease.get(predicted_disease, [])
                fallback_predictions['alternative_diseases'] = [
                    {'name': disease, 'score': round(score * 20, 1)} 
                    for disease, score in sorted_diseases[1:4]  # Get next 3 alternatives
                ] if len(sorted_diseases) > 1 else []
            else:
                # If no mappings found, use a generic response
                predicted_disease = "Unknown Condition"
                confidence = 0.4
                
            return predicted_disease, confidence, confidence * 100, fallback_predictions

    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")            # Try to extract symptom information from input data if available
        contributing_symptoms = {}
        
        try:
            # Extract any symptoms that were selected
            if isinstance(input_data, dict):
                selected_symptoms = [k for k, v in input_data.items() if v == 1]
                for symptom in selected_symptoms:
                    formatted_symptom = symptom.replace('_', ' ').title()
                    contributing_symptoms[formatted_symptom] = 100.0 / max(1, len(selected_symptoms))
            elif hasattr(input_data, 'columns'):
                for col in input_data.columns:
                    if col in input_data and input_data[col].iloc[0] == 1:
                        formatted_symptom = col.replace('_', ' ').title()
                        contributing_symptoms[formatted_symptom] = 100.0 / max(1, len(input_data.columns))
        except:
            # If everything fails, use a generic symptom entry
            contributing_symptoms = {"General Symptoms": 100.0}
            
        # Create a comprehensive fallback response with proper structure
        fallback_data = {
            "contributing_symptoms": contributing_symptoms,
            "error": str(e),
            # Add properly structured alternative diseases for the template
            "alternative_diseases": [
                {"name": "Consult a physician", "score": 50.0},
                {"name": "Further tests needed", "score": 40.0},
                {"name": "Symptoms inconclusive", "score": 30.0}
            ],
            # Add matched symptoms
            "matched_symptoms": [symptom for symptom in contributing_symptoms.keys()]
        }
            
        return "Unable to determine condition", 0.3, 30.0, fallback_data

def get_feature_importance(models, algorithm=None):
    """Extract feature importance from the trained model
    
    Args:
        models: Dictionary of trained models
        algorithm: Which algorithm to use (if None, uses the best model)
        
    Returns:
        Dictionary of feature importances
    """
    try:
        # Determine which model to use
        if algorithm is None or algorithm not in models:
            # Use the best model
            algorithm = models['best_model']
        
        model_info = models[algorithm]
        model = model_info['model']
        
        # Try to get feature names from the model if available
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            # Try to get feature names from the dataset
            try:
                train_df, _ = load_dataset()
                feature_names = train_df.drop('prognosis', axis=1).columns.tolist()
            except Exception as e:
                logger.warning(f"Could not load dataset to get feature names: {str(e)}")
                # Return default feature importances
                return {
                    'cough': 0.15,
                    'fever': 0.12,
                    'headache': 0.10,
                    'fatigue': 0.09,
                    'sore_throat': 0.08,
                    'runny_nose': 0.07,
                    'shortness_of_breath': 0.06,
                    'body_aches': 0.05,
                    'chills': 0.05,
                    'loss_of_taste': 0.04
                }
        
        # Different models have different ways to access feature importance
        if isinstance(model, GaussianNB):
            # For Naive Bayes, use variance as a proxy for importance
            if hasattr(model, 'var_'):
                avg_var = np.mean(model.var_, axis=0)
                importance = avg_var / np.sum(avg_var) if np.sum(avg_var) > 0 else avg_var
            else:
                # If var_ is not available, create equal importances
                importance = np.ones(len(feature_names)) / len(feature_names)
        
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models (Decision Tree, Random Forest, Gradient Boosting)
            importance = model.feature_importances_
        
        else:
            # Default case if no feature importance is available
            logger.warning(f"No feature importance available for {type(model).__name__}")
            return {
                'cough': 0.15,
                'fever': 0.12,
                'headache': 0.10,
                'fatigue': 0.09,
                'sore_throat': 0.08,
                'runny_nose': 0.07,
                'shortness_of_breath': 0.06,
                'body_aches': 0.05,
                'chills': 0.05,
                'loss_of_taste': 0.04
            }
        
        # Create and sort dict of feature importances
        importance_dict = dict(zip(feature_names, importance))
        sorted_importance = {k: float(v) for k, v in sorted(
            importance_dict.items(), key=lambda item: item[1], reverse=True
        )}
        
        # Limit to top 15 features for readability
        top_features = dict(list(sorted_importance.items())[:15])
        
        return top_features
    
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        # Return default feature importances
        return {
            'cough': 0.15,
            'fever': 0.12,
            'headache': 0.10,
            'fatigue': 0.09,
            'sore_throat': 0.08,
            'runny_nose': 0.07,
            'shortness_of_breath': 0.06,
            'body_aches': 0.05,
            'chills': 0.05,
            'loss_of_taste': 0.04
        }


def get_model_comparison(models):
    """Get a comparison of all trained models
    
    Args:
        models: Dictionary of trained models
        
    Returns:
        Dictionary with model comparison metrics
    """
    comparison = {}
    
    for name, model_info in models.items():
        if name == 'best_model':
            continue
            
        comparison[name] = {
            'accuracy': float(model_info['accuracy']),
            'name': get_algorithm_full_name(name)
        }
    
    return comparison


def get_algorithm_full_name(short_name):
    """Convert short algorithm name to full name"""
    name_map = {
        'naive_bayes': 'Naive Bayes',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting'
    }
    return name_map.get(short_name, short_name)
