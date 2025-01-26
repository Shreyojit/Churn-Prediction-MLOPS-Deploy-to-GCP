import streamlit as st
import requests
import os
from dotenv import load_dotenv
import json





# Load environment variables from .env file
load_dotenv()

# Set the FastAPI URL dynamically
FASTAPI_URL = "http://fastapi:80/predict"  # Use Docker service name

# Page title and description
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("Fraud Detection System")
st.write("This application predicts the likelihood of fraud for a given transaction using a pre-trained model.")

# Input fields for the transaction details
st.header("Input Transaction Details")
transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
)
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01, format="%.2f")
name_dest = st.text_input("Destination Account Name", "C123456789")
old_balance_org = st.number_input("Original Account Balance (Before)", min_value=0.0, step=0.01, format="%.2f")
new_balance_orig = st.number_input("Original Account Balance (After)", min_value=0.0, step=0.01, format="%.2f")
old_balance_dest = st.number_input("Destination Account Balance (Before)", min_value=0.0, step=0.01, format="%.2f")
new_balance_dest = st.number_input("Destination Account Balance (After)", min_value=0.0, step=0.01, format="%.2f")

# Predict button
if st.button("Predict Fraud"):
    # Prepare the data as a dictionary
    transaction_data = {
        "type": transaction_type,
        "amount": amount,
        "nameDest": name_dest,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": new_balance_orig,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
    }

    # Display loading spinner during prediction
    with st.spinner("Processing..."):
        try:
            # Make a POST request to the FastAPI endpoint
            response = requests.post(FASTAPI_URL, json=transaction_data)
            if response.status_code == 200:
                result = response.json()
                fraud_probability = result.get("fraud_probability", None)
                st.success(f"Fraud Probability: {fraud_probability * 100:.2f}%")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center;'>
        Built with ❤️ using Streamlit and FastAPI | © 2025
    </p>
    """,
    unsafe_allow_html=True
)