from app import app, db
from models import User, HealthRecord, SymptomRecord

# Create database tables
with app.app_context():
    db.create_all()
    print("Database tables created successfully.")
    print(f"Created tables: {db.metadata.tables.keys()}")
