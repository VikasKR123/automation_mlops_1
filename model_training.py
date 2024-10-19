import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data import generate_data
import numpy as np

# Load the initial reference data
reference_data = generate_data(num_samples=100, drift=False)
target = np.random.randint(0, 2, size=100)  # Random binary target for classification

# Train and log the model
def train_and_log_model(reference_data, target):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(reference_data, target)
    accuracy = accuracy_score(target, model.predict(reference_data))
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        model_uri = mlflow.get_artifact_uri("model")  # Get the model URI
    
    return model_uri

# Log the reference model for rollback purposes
reference_model_uri = train_and_log_model(reference_data, target)
print(f"Initial model saved at: {reference_model_uri}")
