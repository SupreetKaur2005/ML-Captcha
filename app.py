from flask import Flask, request, jsonify, send_from_directory
import os
import json
import logging
import numpy as np
import joblib

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths to the models and features
MODEL_RF_PATH = "models/optimized_rf_classifier.pkl"
MODEL_GB_PATH = "models/gradient_boosting_classifier.pkl"
FEATURES_PATH = "models/model_features.txt"

# Load the trained models and required features
try:
    model_rf = joblib.load(MODEL_RF_PATH)
    model_gb = joblib.load(MODEL_GB_PATH)
    with open(FEATURES_PATH, "r") as f:
        required_features = [line.strip() for line in f.readlines()]
    logging.info("✅ Models and feature list loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"⚠️ Model or feature list not found: {e}")
    model_rf = None
    model_gb = None
    required_features = []

# -------------------------------
# Enhanced Feature Extraction
# -------------------------------
def extract_features(data):
    """Extract features from the input behavioral data."""
    try:
        mouse_data = data.get("mouseEvents", [])
        if not mouse_data or len(mouse_data) < 2:
            return {"movement_count": 0, "total_distance": 0.0, "avg_speed": 0.0}

        # Extract positions and timestamps
        positions = [(e["x"], e["y"]) for e in mouse_data]
        timestamps = [e["timestamp"] for e in mouse_data]

        # Calculate distances, speeds, and additional metrics
        distances = [np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i]))
                     for i in range(len(positions) - 1)]
        deltas = [(timestamps[i+1] - timestamps[i]) for i in range(len(timestamps) - 1)]

        speeds = [d / t if t > 0 else 0 for d, t in zip(distances, deltas)]

        return {
            "movement_count": len(mouse_data),
            "total_distance": sum(distances),
            "avg_speed": np.mean(speeds),
            "max_speed": max(speeds),
            "duration_ms": timestamps[-1] - timestamps[0],
        }
    except Exception as e:
        logging.error(f"Feature extraction error: {e}")
        return {}

# -------------------------------
# API Route: Collect & Predict
# -------------------------------
@app.route('/api/captcha-data', methods=['POST'])
def collect_data():
    """API endpoint to collect behavioral data and make predictions."""
    if not model_rf or not model_gb:
        return jsonify({"error": "Models are not loaded. Please train the models first."}), 500

    data = request.json

    # Extract features
    extracted_features = extract_features(data)
    if not extracted_features:
        return jsonify({"error": "Feature extraction failed."}), 400

    # Prepare data for prediction
    feature_vector = np.array([extracted_features.get(f, 0) for f in required_features]).reshape(1, -1)

    # Predictions from both models
    try:
        prediction_rf = model_rf.predict(feature_vector)[0]
        prediction_gb = model_gb.predict(feature_vector)[0]
        confidence_rf = model_rf.predict_proba(feature_vector)[0].max()
        confidence_gb = model_gb.predict_proba(feature_vector)[0].max()
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed."}), 500

    # Combine predictions (simple majority voting)
    final_prediction = (prediction_rf + prediction_gb) // 2

    # Prepare response
    response = {
        "final_prediction": "bot" if final_prediction == 1 else "human",
        "model_confidences": {
            "random_forest": confidence_rf,
            "gradient_boosting": confidence_gb
        },
        "individual_predictions": {
            "random_forest": "bot" if prediction_rf == 1 else "human",
            "gradient_boosting": "bot" if prediction_gb == 1 else "human"
        },
        "extracted_features": extracted_features
    }

    return jsonify(response)

# -------------------------------
# API Route: Home
# -------------------------------
@app.route('/')
def home():
    """Serve the homepage."""
    return send_from_directory('.', 'index.html')

# -------------------------------
# Main
# -------------------------------
if __name__ == '__main__':
    try:
        logging.info("Starting app.py...")
        app.run(debug=True)
    except KeyboardInterrupt:
        logging.info("Shutting down app.py...")
