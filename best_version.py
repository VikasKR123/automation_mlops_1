# import mlflow
# from mlflow.tracking import MlflowClient

# client = MlflowClient()

# # List all models and versions by querying for model versions
# model_name = "Best_Model"  # Replace with your actual model name

# versions = client.search_model_versions(f"name='{model_name}'")
# for version in versions:
#     print(f"Model Name: {version.name}, Version: {version.version}, Stage: {version.current_stage}")
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the latest version of 'Best_Model' not in any stage
latest_version = client.get_latest_versions("Best_Model", stages=['None'])[0].version

# Transition the model version to the 'Production' stage
client.transition_model_version_stage(
    name="Best_Model",
    version=latest_version,
    stage="Production"
)

print(f"Model 'Best_Model' version {latest_version} transitioned to 'Production' stage.")
