import pandas as pd
from utils import train_models, predict_health_condition, load_dataset

# Load the dataset
try:
    train_df, test_df = load_dataset()
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit()

# Train the models
try:
    models, preprocessor = train_models(train_df, test_df, algorithm='all')
    print("Models trained successfully.")
except Exception as e:
    print(f"Error training models: {e}")
    exit()

# Test the prediction pipeline
input_data = {
    'itching': 1,
    'skin_rash': 1,
    'nodal_skin_eruptions': 0,
    'continuous_sneezing': 0,
    'shivering': 0,
    'chills': 0,
    'joint_pain': 0,
    'stomach_pain': 0,
    'acidity': 0,
    'ulcers_on_tongue': 0
}

try:
    prediction, probability, confidence_score, _ = predict_health_condition(models, preprocessor, input_data, algorithm='naive_bayes')
    print(f"Prediction: {prediction}, Probability: {probability}, Confidence Score: {confidence_score}")
except Exception as e:
    print(f"Error in prediction: {e}")
