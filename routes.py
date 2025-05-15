import os
import pandas as pd
from flask import render_template, request, redirect, url_for, flash, jsonify, session
from app import app, db
from models import HealthRecord
from utils import train_model, predict_health_risk, load_dataset, get_feature_importance

# Initialize and train the model
model, vectorizer = None, None

def initialize_model():
    global model, vectorizer
    df = load_dataset()
    model, vectorizer = train_model(df)
    app.logger.info("Model trained successfully")

# Initialize the model on startup
with app.app_context():
    initialize_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        
        # Create a health record object
        record = HealthRecord()
        # Set the fields manually to avoid constructor errors
        record.user_id = 1  # Default user ID
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
        
        # Run prediction
        global model, vectorizer
        if model is None or vectorizer is None:
            df = load_dataset()
            model, vectorizer = train_model(df)
            
        prediction, probability, risk_score = predict_health_risk(model, vectorizer, input_data)
        
        # Get feature importance for visualization
        features_importance = get_feature_importance(model, vectorizer)
        
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
        
        return redirect(url_for('results'))
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/results')
def results():
    prediction = session.get('prediction', None)
    user_input = session.get('user_input', None)
    
    if not prediction:
        flash("No prediction data available. Please submit the form first.", "warning")
        return redirect(url_for('index'))
    
    return render_template('results.html', 
                           prediction=prediction, 
                           user_input=user_input)

@app.route('/api/health-check')
def health_check():
    """API endpoint to check if the service is running"""
    return jsonify({"status": "healthy", "message": "Health detection system is operational"})
