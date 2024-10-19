import time
import mlflow
from drift_detection import detect_drift
from data import generate_data
from model_training import train_and_log_model
import numpy as np

# Reference data (initial state, without drift)
reference_data = generate_data(num_samples=100, drift=False)

# Train and log the initial model if not already trained
try:
    reference_model_uri = mlflow.register_model("models:/Best_Model/Production")
except Exception as e:
    print(f"No existing model found, training the initial model: {e}")
    target = np.random.randint(0, 2, size=100)
    reference_model_uri = train_and_log_model(reference_data, target)
    mlflow.register_model(reference_model_uri, "Best_Model")

# Monitoring loop to check for drift and rollback if drift is detected
while True:
    # Simulate incoming new data (without drift for some iterations, then with drift)
    new_data = generate_data(num_samples=100, drift=True)  # Simulate drift after some time
    
    # Detect drift
    drift_detected = detect_drift(new_data, reference_data)
    
    if drift_detected:
        print("Drift detected! Rolling back to the reference model...")
        # Rollback logic (re-register the reference model)
        mlflow.register_model(reference_model_uri, "Best_Model")
    

    time.sleep(30)  
