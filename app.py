from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

model_uri = "models:/Best_Model/Production"

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully from MLflow")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        # Get input data from request body
        input_data = request.json

        # Ensure input data is in a list format
        if not isinstance(input_data, list):
            input_data = [input_data]

        # Convert input data into a pandas DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Make predictions using the loaded model
        prediction = model.predict(input_df)
        
        # Return predictions as a JSON response
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

