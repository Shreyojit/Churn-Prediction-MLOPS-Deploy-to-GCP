# Use an official Python runtime as a parent image
FROM python:3.10.16-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY app/requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-dotenv

# Copy the contents of the app directory into the container at /app
COPY app/ /app/

# Expose ports for FastAPI and Streamlit
EXPOSE 80 8501