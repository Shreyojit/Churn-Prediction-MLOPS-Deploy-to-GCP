import os
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
    return prediction[0, 0]  # Return the first prediction (fraud probability)

# Function to load JSON file and get prediction
def predict_from_json(json_file_path):
    """
    Load a JSON file, preprocess the data, and predict fraud likelihood.

    Parameters:
        json_file_path (str): Path to the JSON file containing input data.

    Returns:
        float: Predicted fraud probability.
    """
    # Load JSON file
    with open(json_file_path, 'r') as file:
        json_input = json.load(file)
    
    # Get prediction
    fraud_probability = predict_fraud(json_input)
    print(f"Fraud probability: {fraud_probability}")
    return fraud_probability

# Example usage
if __name__ == "__main__":
    # Path to the JSON file containing input data
    json_file_path = os.path.join(base_dir, "input_data.json")
    
    # Check if the JSON file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found at {json_file_path}")
    
    # Get prediction
    fraud_probability = predict_from_json(json_file_path)
    print(f"Final Fraud Probability: {fraud_probability}")