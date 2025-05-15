from app import db
from flask_login import UserMixin
from datetime import datetime

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    health_records = db.relationship('HealthRecord', backref='user', lazy=True)

class HealthRecord(db.Model):
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
