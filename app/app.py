import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import json

# Define the base directory (where the script is located)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths relative to the base directory
model_path = os.path.join(base_dir, "model", "fraud_detection_model.h5")
scaler_path = os.path.join(base_dir, "model", "scaler.pkl")
training_info_path = os.path.join(base_dir, "model", "training_info.pkl")


print(f"Model Path: {model_path}")
print(f"Scaler Path: {scaler_path}")
print(f"Training Info Path: {training_info_path}")




# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
else:
    # Load the trained model, scaler, and training info
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Loading scaler...")
    scaler = joblib.load(scaler_path)
    print("Loading training info...")
    training_info = joblib.load(training_info_path)

# Define FastAPI app
app = FastAPI()

# Pydantic model for input validation
class FraudPredictionRequest(BaseModel):
    type: str
    amount: float
    nameDest: str
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

# Function to process input data and predict fraud likelihood
def preprocess_input(json_input):
    """
    Preprocess a single JSON or dictionary input for prediction.

    Parameters:
        json_input (dict): Input data in JSON or dictionary format.

    Returns:
        np.ndarray: Preprocessed input ready for prediction.
    """
    # Convert to DataFrame for easier processing
    df = pd.DataFrame([json_input])

    # One-hot encode the 'type' column
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'], prefix='tp')

    # Ensure all one-hot encoded columns match training data
    for col in training_info['columns']:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value 0
    df = df[training_info['columns']]  # Reorder columns to match training order

    # Handle 'nameDest' column if present
    if 'nameDest' in df.columns:
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(df['nameDest'])
        customers = tokenizer.texts_to_sequences(df['nameDest'])
        customers = tf.keras.preprocessing.sequence.pad_sequences(customers, maxlen=1)
        df['customer'] = np.squeeze(customers)
        df = df.drop('nameDest', axis=1)

    # Standardize features using the saved scaler
    df = pd.DataFrame(scaler.transform(df), columns=training_info['columns'])

    return df.values  # Return as NumPy array for prediction

# Function to predict fraud likelihood
def predict_fraud(json_input):
    """
    Predict fraud likelihood for a single input.

    Parameters:
        json_input (dict): Input data in JSON or dictionary format.

    Returns:
        float: Predicted fraud probability.
    """
    print("Predicting fraud...")  # Debugging statement
    
    # Process the input data before making the prediction
    processed_data = preprocess_input(json_input)
    
    print("Model prediction in progress...")  # Debugging statement
    prediction = model.predict(processed_data)
    
    print("Prediction complete.")  # Debugging statement
    return float(prediction[0, 0])  # Convert numpy.float32 to Python float

@app.get("/")
def read_root():
    return {"message": "Hello World"}


# FastAPI endpoint for fraud prediction
@app.post("/predict")
async def predict(request: FraudPredictionRequest):
    try:
        fraud_probability = predict_fraud(request.dict())
        return {"fraud_probability": fraud_probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Fraud Detection API"}