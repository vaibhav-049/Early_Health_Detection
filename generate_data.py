import csv
import random
import pandas as pd
import numpy as np

# Read the existing dataset to understand the patterns
df = pd.read_csv('data/sample_health_data.csv')

# Define ranges and distributions based on the existing data
age_range = (25, 75)
heart_rate_range = (60, 100)
bp_systolic_range = (100, 180)
bp_diastolic_range = (60, 110)
cholesterol_range = (150, 300)
blood_sugar_range = (70, 150)
bmi_range = (18.5, 35.0)

# Function to generate realistic health data
def generate_health_data(num_records=1000):
    data = []
    
    # Keep the existing data
    for _, row in df.iterrows():
        data.append(dict(row))
    
    # Generate additional records
    additional_records = num_records - len(df)
    
    for _ in range(additional_records):
        age = random.randint(age_range[0], age_range[1])
        gender = random.choice(['male', 'female'])
        
        # Generate health metrics with some correlation to age
        heart_rate = max(60, min(100, random.normalvariate(75, 8) + (age - 40) * 0.1))
        bp_systolic = max(100, min(180, random.normalvariate(120, 15) + age * 0.5))
        bp_diastolic = max(60, min(110, random.normalvariate(80, 10) + age * 0.3))
        cholesterol = max(150, min(300, random.normalvariate(200, 30) + age * 1.2))
        blood_sugar = max(70, min(150, random.normalvariate(90, 15) + age * 0.6))
        bmi = max(18.5, min(35.0, random.normalvariate(24.0, 3.5) + (age - 40) * 0.05))
        
        # Lifestyle factors
        smoking = np.random.choice([0, 1], p=[0.7, 0.3])
        alcohol_consumption = np.random.choice([0, 1], p=[0.6, 0.4])
        physical_activity = np.random.choice(range(7), p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1])
        family_history = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Determine risk level based on factors
        risk_score = 0
        risk_score += (age > 50) * 2
        risk_score += (bp_systolic > 140 or bp_diastolic > 90) * 2
        risk_score += (cholesterol > 240) * 2
        risk_score += (blood_sugar > 110) * 1
        risk_score += (bmi > 27) * 1
        risk_score += smoking * 2
        risk_score += alcohol_consumption * 1
        risk_score += (physical_activity < 3) * 1
        risk_score += family_history * 2
        
        if risk_score < 4:
            risk_level = 'low'
        elif risk_score < 8:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        record = {
            'age': int(age),
            'gender': gender,
            'heart_rate': int(heart_rate),
            'blood_pressure_systolic': int(bp_systolic),
            'blood_pressure_diastolic': int(bp_diastolic),
            'cholesterol': int(cholesterol),
            'blood_sugar': int(blood_sugar),
            'bmi': round(bmi, 1),
            'smoking': smoking,
            'alcohol_consumption': alcohol_consumption,
            'physical_activity': physical_activity,
            'family_history': family_history,
            'risk_level': risk_level
        }
        
        data.append(record)
    
    return data

# Generate the data
health_data = generate_health_data(1000)

# Write to CSV
with open('data/sample_health_data.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=health_data[0].keys())
    writer.writeheader()
    writer.writerows(health_data)

print(f"Generated {len(health_data)} health records and saved to data/sample_health_data.csv")
