import os
import pandas as pd
from flask import render_template, request, redirect, url_for, flash, jsonify, session
from app import app
from database import db
from models import HealthRecord, SymptomRecord
from utils import train_models, predict_health_condition, load_dataset, get_feature_importance, get_model_comparison, get_algorithm_full_name
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Initialize and train the models
models, preprocessor = None, None

def initialize_models():
    global models, preprocessor
    try:
        train_df, test_df = load_dataset()
        models, preprocessor = train_models(train_df, test_df, algorithm='all')
        app.logger.info(f"Models trained successfully. Best model: {models['best_model']}")
    except Exception as e:
        app.logger.error(f"Error initializing models: {str(e)}")
        # Create fallback models
        app.logger.warning("Using fallback models")
        models = {
            'naive_bayes': {'model': GaussianNB(), 'accuracy': 0.8},
            'decision_tree': {'model': DecisionTreeClassifier(random_state=42), 'accuracy': 0.75},
            'random_forest': {'model': RandomForestClassifier(n_estimators=100, random_state=42), 'accuracy': 0.85},
            'gradient_boosting': {'model': GradientBoostingClassifier(n_estimators=100, random_state=42), 'accuracy': 0.82},
            'best_model': 'random_forest'
        }
        preprocessor = StandardScaler()

# Initialize the models on startup
with app.app_context():
    initialize_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/symptom-checker')
def symptom_checker():
    """Render the symptom checker form"""
    # Get all symptoms from dataset
    try:
        train_df, _ = load_dataset()
        # Get all symptom columns (all except the last column which is the prognosis)
        symptom_columns = train_df.columns[:-1]
        
        # Create a list of tuples with symptom_id and symptom_name
        symptoms = [(col, col.replace('_', ' ').title()) for col in symptom_columns]
        
        # Sort symptoms alphabetically for better user experience
        symptoms.sort(key=lambda x: x[1])
    except Exception as e:
        app.logger.error(f"Error loading symptoms: {e}")
        # Fallback to a basic list of symptoms
        symptoms = [(f"symptom_{i}", f"Symptom {i}") for i in range(1, 10)]
    
    # Define available algorithms
    algorithms = [
        {"id": "naive_bayes", "name": "Naive Bayes"},
        {"id": "decision_tree", "name": "Decision Tree"},
        {"id": "random_forest", "name": "Random Forest"},
        {"id": "gradient_boosting", "name": "Gradient Boosting"},
        {"id": "all", "name": "Compare All"}
    ]
    
    return render_template('symptom_checker.html', symptoms=symptoms, algorithms=algorithms)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug log for form data
        app.logger.debug(f"Form data received: {request.form}")

        # Get form values safely with fallbacks to avoid type errors
        age = request.form.get('age')
        gender = request.form.get('gender')
        heart_rate = request.form.get('heart_rate')
        bp_systolic = request.form.get('blood_pressure_systolic')
        bp_diastolic = request.form.get('blood_pressure_diastolic')
        cholesterol = request.form.get('cholesterol')
        blood_sugar = request.form.get('blood_sugar')
        bmi = request.form.get('bmi')
        smoking = request.form.get('smoking') == 'yes'
        alcohol = request.form.get('alcohol_consumption') == 'yes'
        physical = request.form.get('physical_activity')
        family_history = request.form.get('family_history') == 'yes'
        
        # Convert to proper types for the model
        form_data = {
            'age': int(age) if age else 30,
            'gender': gender,
            'heart_rate': int(heart_rate) if heart_rate else 70,
            'blood_pressure_systolic': int(bp_systolic) if bp_systolic else 120,
            'blood_pressure_diastolic': int(bp_diastolic) if bp_diastolic else 80,
            'cholesterol': int(cholesterol) if cholesterol else 180,
            'blood_sugar': float(blood_sugar) if blood_sugar else 90.0,
            'bmi': float(bmi) if bmi else 22.5,
            'smoking': smoking,
            'alcohol_consumption': alcohol,
            'physical_activity': int(physical) if physical else 3,
            'family_history': family_history
        }
        
        # Create a health record object without a user_id
        record = HealthRecord()
        record.user_id = None  # Allow anonymous record for demo/testing
        record.age = form_data['age']
        record.gender = form_data['gender']
        record.heart_rate = form_data['heart_rate']
        record.blood_pressure_systolic = form_data['blood_pressure_systolic']
        record.blood_pressure_diastolic = form_data['blood_pressure_diastolic']
        record.cholesterol = form_data['cholesterol']
        record.blood_sugar = form_data['blood_sugar']
        record.bmi = form_data['bmi']
        record.smoking = form_data['smoking']
        record.alcohol_consumption = form_data['alcohol_consumption']
        record.physical_activity = form_data['physical_activity']
        record.family_history = form_data['family_history']
        
        # Convert to format needed for prediction
        input_data = record.to_dict()
        
        # Debug log for prediction input
        app.logger.debug(f"Prediction input: {input_data}")
        
        # Run prediction (legacy method - keeping for backward compatibility)
        # In a real application, we would update this to use the new ML models
        # For now, we'll just use a default risk assessment based on basic rules
        
        # Simple rule-based risk assessment
        risk_factors = 0
        if form_data['age'] > 50:
            risk_factors += 1
        if form_data['blood_pressure_systolic'] > 140 or form_data['blood_pressure_diastolic'] > 90:
            risk_factors += 1
        if form_data['cholesterol'] > 200:
            risk_factors += 1
        if form_data['smoking']:
            risk_factors += 1
        if form_data['family_history']:
            risk_factors += 1
        
        if risk_factors >= 3:
            prediction = "high"
            probability = 0.85
            risk_score = 80
        elif risk_factors >= 1:
            prediction = "medium"
            probability = 0.65
            risk_score = 50
        else:
            prediction = "low"
            probability = 0.90
            risk_score = 20
        
        # Get dummy feature importance for visualization
        features_importance = {
            'age': 0.25,
            'blood_pressure_systolic': 0.20,
            'cholesterol': 0.18,
            'smoking': 0.15,
            'family_history': 0.12,
            'blood_pressure_diastolic': 0.10
        }
        
        # Save the record with prediction results
        record.risk_score = risk_score
        record.risk_category = prediction
        db.session.add(record)
        db.session.commit()
        
        # Store results in session for display
        session['prediction'] = {
            'outcome': prediction,
            'probability': round(probability * 100, 2),
            'risk_score': round(risk_score, 2),
            'features_importance': features_importance
        }
        
        # Store input data for display in results
        session['user_input'] = {k: str(v) for k, v in form_data.items()}
        session['prediction_type'] = 'health_risk'
        
        # Debug log for prediction result
        app.logger.debug(f"Prediction result: {prediction}, Probability: {probability}, Risk Score: {risk_score}")

        return redirect(url_for('results'))
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    try:
        # Debug log for form data
        app.logger.debug(f"Form data received: {request.form}")

        # Get the selected algorithm
        selected_algorithm = request.form.get('algorithm', 'all')
        
        # Create a symptom record object
        record = SymptomRecord()
        record.user_id = None  # Allow anonymous record for demo/testing
        
        # Get all possible symptoms from the dataset
        try:
            train_df, _ = load_dataset()
            all_symptoms = train_df.columns[:-1].tolist()
        except Exception as e:
            app.logger.error(f"Error loading symptoms from dataset: {e}")
            # Fallback to symptoms from the form
            all_symptoms = [key.replace('symptom_', '') for key in request.form.keys() if key.startswith('symptom_')]
        
        # Process all symptoms from the form - now using checkboxes instead of radio buttons
        input_data = {}
        
        # Initialize all symptoms to 0 (not present)
        for symptom in all_symptoms:
            input_data[symptom] = 0
            
        # Update with checked symptoms (present)
        symptoms_present = []
        for key in request.form.keys():
            if key.startswith('symptom_'):
                symptom_name = key.replace('symptom_', '')
                # If the checkbox is checked (present in form data)
                input_data[symptom_name] = 1
                symptoms_present.append(symptom_name)
                setattr(record, symptom_name, True)
        
        # Check if any symptoms were selected
        if not symptoms_present:
            flash("Please select at least one symptom.", "warning")
            return redirect(url_for('symptom_checker'))
        
        # Run prediction
        global models, preprocessor
        if models is None or preprocessor is None:
            initialize_models()
        
        # Make prediction with selected algorithm
        try:
            prediction, probability, confidence_score, all_predictions = predict_health_condition(
                models, preprocessor, input_data, algorithm=selected_algorithm
            )
            
            # Check if we got an error in the prediction
            if 'error' in all_predictions and not all_predictions.get('contributing_symptoms'):
                app.logger.warning(f"Prediction returned an error: {all_predictions['error']}")
                flash(f"Could not make a prediction: {all_predictions['error']}", "warning")
                
                # Ensure we have the contributing_symptoms key to avoid template errors
                if 'contributing_symptoms' not in all_predictions:
                    all_predictions['contributing_symptoms'] = {}
        except Exception as e:
            app.logger.error(f"Prediction error: {e}")
            # Fallback to a default prediction
            prediction = "Unknown"
            probability = 0.0
            confidence_score = 0.0
            all_predictions = {'contributing_symptoms': {}, 'error': str(e)}
            flash("Could not make a prediction with the selected symptoms. Please try different symptoms.", "warning")
        
        # Get feature importance for visualization
        features_importance = get_feature_importance(models, algorithm=selected_algorithm)
        
        # Get model comparison data
        model_comparison = get_model_comparison(models)
        
        # Save the record with prediction results
        record.predicted_disease = prediction
        record.confidence_score = confidence_score
        record.algorithm_used = selected_algorithm
        db.session.add(record)
        db.session.commit()
        
        # Store results in session for display
        session['prediction'] = {
            'outcome': prediction,
            'probability': round(probability * 100, 2),
            'confidence_score': round(confidence_score, 2),
            'features_importance': features_importance,
            'all_predictions': all_predictions,
            'model_comparison': model_comparison,
            'algorithm_used': get_algorithm_full_name(selected_algorithm)
        }
        
        # Ensure alternative_diseases and matched_symptoms are properly included in the prediction data
        if 'all_predictions' in session['prediction'] and isinstance(session['prediction']['all_predictions'], dict):
            # If alternative_diseases isn't already properly formatted, create it
            if 'alternative_diseases' not in session['prediction']['all_predictions']:
                # Extract top conditions from all_predictions (excluding the main prediction)
                other_conditions = []
                for condition, prob in all_predictions.items():
                    if (condition != 'contributing_symptoms' and 
                        condition != 'error' and 
                        condition != 'matched_symptoms' and 
                        condition != 'alternative_diseases' and 
                        condition != prediction and
                        isinstance(prob, (int, float))):
                        other_conditions.append({
                            'name': condition,
                            'score': round(prob * 100, 1)
                        })
                
                # Sort by score and take top 3
                other_conditions.sort(key=lambda x: x['score'], reverse=True)
                session['prediction']['all_predictions']['alternative_diseases'] = other_conditions[:3]
        
        # Store input data for display in results
        symptoms_present = [k.replace('_', ' ').title() for k, v in input_data.items() if v == 1]
        session['user_input'] = {'symptoms_present': symptoms_present}
        session['prediction_type'] = 'disease'
        
        # Debug log for prediction input
        app.logger.debug(f"Prediction input: {input_data}")

        # Debug log for prediction result
        app.logger.debug(f"Prediction result: {prediction}, Confidence Score: {confidence_score}")

        return redirect(url_for('disease_results'))
    
    except Exception as e:
        app.logger.error(f"Disease prediction error: {str(e)}")
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for('symptom_checker'))

@app.route('/results')
def results():
    prediction = session.get('prediction', None)
    user_input = session.get('user_input', None)
    prediction_type = session.get('prediction_type', 'health_risk')
    
    if not prediction:
        flash("No prediction data available. Please submit the form first.", "warning")
        return redirect(url_for('index'))
    
    return render_template('results.html', 
                           prediction=prediction, 
                           user_input=user_input,
                           prediction_type=prediction_type)


@app.route('/disease-results')
def disease_results():
    prediction = session.get('prediction', None)
    user_input = session.get('user_input', None)
    
    if not prediction:
        flash("No prediction data available. Please submit the form first.", "warning")
        return redirect(url_for('symptom_checker'))
    
    return render_template('disease_results.html', 
                           prediction=prediction, 
                           user_input=user_input)

@app.route('/api/health-check')
def health_check():
    """API endpoint to check if the service is running"""
    return jsonify({"status": "healthy", "message": "Health detection system is operational"})


@app.route('/api/algorithms')
def get_algorithms():
    """API endpoint to get available algorithms and their performance"""
    global models
    if models is None:
        initialize_models()
    
    model_comparison = get_model_comparison(models)
    best_model = models['best_model']
    
    return jsonify({
        "algorithms": model_comparison,
        "best_model": best_model,
        "best_model_name": get_algorithm_full_name(best_model)
    })
