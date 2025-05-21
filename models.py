from flask_login import UserMixin
from datetime import datetime
from database import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    health_records = db.relationship('HealthRecord', backref='user', lazy=True)
    symptom_records = db.relationship('SymptomRecord', backref='user', lazy=True)

class HealthRecord(db.Model):
    """Original health record model for cardiovascular risk assessment"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='SET NULL'), nullable=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    heart_rate = db.Column(db.Integer, nullable=False)
    blood_pressure_systolic = db.Column(db.Integer, nullable=False)
    blood_pressure_diastolic = db.Column(db.Integer, nullable=False)
    cholesterol = db.Column(db.Integer, nullable=False)
    blood_sugar = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    smoking = db.Column(db.Boolean, default=False)
    alcohol_consumption = db.Column(db.Boolean, default=False)
    physical_activity = db.Column(db.Integer, nullable=False)  # hours per week
    family_history = db.Column(db.Boolean, default=False)
    risk_score = db.Column(db.Float, nullable=True)  # Risk score calculated by model
    risk_category = db.Column(db.String(20), nullable=True)  # Low, Medium, High
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert record to dictionary for model prediction"""
        return {
            'age': self.age,
            'gender': 1 if self.gender.lower() == 'male' else 0,  # Convert to binary for model
            'heart_rate': self.heart_rate,
            'blood_pressure_systolic': self.blood_pressure_systolic,
            'blood_pressure_diastolic': self.blood_pressure_diastolic,
            'cholesterol': self.cholesterol,
            'blood_sugar': self.blood_sugar,
            'bmi': self.bmi,
            'smoking': 1 if self.smoking else 0,
            'alcohol_consumption': 1 if self.alcohol_consumption else 0,
            'physical_activity': self.physical_activity,
            'family_history': 1 if self.family_history else 0
        }


class SymptomRecord(db.Model):
    """New model for symptom-based disease prediction"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='SET NULL'), nullable=True)
    
    # Common symptoms from the dataset
    itching = db.Column(db.Boolean, default=False)
    skin_rash = db.Column(db.Boolean, default=False)
    nodal_skin_eruptions = db.Column(db.Boolean, default=False)
    continuous_sneezing = db.Column(db.Boolean, default=False)
    shivering = db.Column(db.Boolean, default=False)
    chills = db.Column(db.Boolean, default=False)
    joint_pain = db.Column(db.Boolean, default=False)
    stomach_pain = db.Column(db.Boolean, default=False)
    acidity = db.Column(db.Boolean, default=False)
    ulcers_on_tongue = db.Column(db.Boolean, default=False)
    muscle_wasting = db.Column(db.Boolean, default=False)
    vomiting = db.Column(db.Boolean, default=False)
    burning_micturition = db.Column(db.Boolean, default=False)
    fatigue = db.Column(db.Boolean, default=False)
    weight_gain = db.Column(db.Boolean, default=False)
    anxiety = db.Column(db.Boolean, default=False)
    cold_hands_and_feets = db.Column(db.Boolean, default=False)
    mood_swings = db.Column(db.Boolean, default=False)
    weight_loss = db.Column(db.Boolean, default=False)
    restlessness = db.Column(db.Boolean, default=False)
    lethargy = db.Column(db.Boolean, default=False)
    patches_in_throat = db.Column(db.Boolean, default=False)
    irregular_sugar_level = db.Column(db.Boolean, default=False)
    cough = db.Column(db.Boolean, default=False)
    high_fever = db.Column(db.Boolean, default=False)
    sunken_eyes = db.Column(db.Boolean, default=False)
    breathlessness = db.Column(db.Boolean, default=False)
    sweating = db.Column(db.Boolean, default=False)
    dehydration = db.Column(db.Boolean, default=False)
    indigestion = db.Column(db.Boolean, default=False)
    headache = db.Column(db.Boolean, default=False)
    yellowish_skin = db.Column(db.Boolean, default=False)
    dark_urine = db.Column(db.Boolean, default=False)
    nausea = db.Column(db.Boolean, default=False)
    loss_of_appetite = db.Column(db.Boolean, default=False)
    back_pain = db.Column(db.Boolean, default=False)
    constipation = db.Column(db.Boolean, default=False)
    abdominal_pain = db.Column(db.Boolean, default=False)
    diarrhoea = db.Column(db.Boolean, default=False)
    mild_fever = db.Column(db.Boolean, default=False)
    
    # Prediction results
    predicted_disease = db.Column(db.String(100), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)  # 0-100 scale
    algorithm_used = db.Column(db.String(50), nullable=True)  # Which ML algorithm was used
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert symptom record to dictionary for model prediction"""
        # Get all boolean columns (symptoms)
        symptom_dict = {}
        for column in self.__table__.columns:
            if column.type.python_type == bool and column.name not in ['id', 'user_id', 'created_at']:
                symptom_dict[column.name] = 1 if getattr(self, column.name) else 0
        
        return symptom_dict
