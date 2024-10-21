import time
import mlflow
from drift_detection import detect_drift
from data import generate_data
from model_training import train_and_log_model
import numpy as np

import requests
import json
import os

# Configuration
GITHUB_REPO = "VikasKR123/automation_mlops_1"
GITHUB_TOKEN = os.getenv("TOKEN")  # Get GitHub token from environment variable
WORKFLOW_FILE_NAME = "ci.yml"

# Generate reference data
reference_data = generate_data(num_samples=100, drift=False)

# Function to trigger GitHub Action workflow
def trigger_github_action():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE_NAME}/dispatches"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "ref": "main",  # Branch name where the workflow resides
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 204:
        print("CI/CD triggered successfully.")
    else:
        print(f"Failed to trigger CI/CD: {response.content}")

# Try to get the existing model, or train a new one
try:
    reference_model_uri = mlflow.register_model("models:/Best_Model/Production")
except Exception as e:
    print(f"No existing model found, training the initial model: {e}")
    target = np.random.randint(0, 2, size=100)  # Random binary target
    reference_model_uri = train_and_log_model(reference_data, target)
    mlflow.register_model(reference_model_uri, "Best_Model")

# Monitoring loop
while True:
    new_data = generate_data(num_samples=100, drift=True)  # Simulate drift after some time
    
    # Detect drift
    drift_detected = detect_drift(new_data, reference_data)
    
    if drift_detected:
        print("Drift detected! Rolling back to the reference model...")
        # Rollback logic (re-register the reference model)
        mlflow.register_model(reference_model_uri, "Best_Model")
        print("Drift detected! Triggering CI/CD...")
        trigger_github_action()
    
    time.sleep(30)  # Wait for 30 seconds before checking again
