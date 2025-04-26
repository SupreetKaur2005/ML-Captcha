# from flask import Flask, request, jsonify, send_from_directory
# import os
# import json
# import logging
# import numpy as np
# import joblib

# # Initialize Flask app and logging
# app = Flask(__name__)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Paths to the models and features
# MODEL_RF_PATH = "optimized_rf_classifier.pkl"
# MODEL_GB_PATH = "gradient_boosting_classifier.pkl"
# FEATURES_PATH = "model_features.txt"

# # Load the trained models and required features
# try:
#     model_rf = joblib.load(MODEL_RF_PATH)
#     model_gb = joblib.load(MODEL_GB_PATH)
#     with open(FEATURES_PATH, "r") as f:
#         required_features = [line.strip() for line in f.readlines()]
#     logging.info("✅ Models and feature list loaded successfully.")
# except FileNotFoundError as e:
#     logging.error(f"⚠️ Model or feature list not found: {e}")
#     model_rf = None
#     model_gb = None
#     required_features = []

# # -------------------------------
# # Enhanced Feature Extraction
# # -------------------------------
# def extract_features(data):
#     """Extract features from the input behavioral data."""
#     try:
#         mouse_data = data.get("mouseEvents", [])
#         if not mouse_data or len(mouse_data) < 2:
#             return {"movement_count": 0, "total_distance": 0.0, "avg_speed": 0.0}

#         # Extract positions and timestamps
#         positions = [(e["x"], e["y"]) for e in mouse_data]
#         timestamps = [e["timestamp"] for e in mouse_data]

#         # Calculate distances, speeds, and additional metrics
#         distances = [np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i]))
#                      for i in range(len(positions) - 1)]
#         deltas = [(timestamps[i+1] - timestamps[i]) for i in range(len(timestamps) - 1)]

#         speeds = [d / t if t > 0 else 0 for d, t in zip(distances, deltas)]

#         return {
#             "movement_count": len(mouse_data),
#             "total_distance": sum(distances),
#             "avg_speed": np.mean(speeds),
#             "max_speed": max(speeds),
#             "duration_ms": timestamps[-1] - timestamps[0],
#         }
#     except Exception as e:
#         logging.error(f"Feature extraction error: {e}")
#         return {}

# # -------------------------------
# # API Route: Collect & Predict
# # -------------------------------
# @app.route('/api/captcha-data', methods=['POST'])
# def collect_data():
#     """API endpoint to collect behavioral data and make predictions."""
#     if not model_rf or not model_gb:
#         return jsonify({"error": "Models are not loaded. Please train the models first."}), 500

#     data = request.json

#     # Extract features
#     extracted_features = extract_features(data)
#     if not extracted_features:
#         return jsonify({"error": "Feature extraction failed."}), 400

#     # Prepare data for prediction
#     feature_vector = np.array([extracted_features.get(f, 0) for f in required_features]).reshape(1, -1)

#     # Predictions from both models
#     try:
#         prediction_rf = model_rf.predict(feature_vector)[0]
#         prediction_gb = model_gb.predict(feature_vector)[0]
#         confidence_rf = model_rf.predict_proba(feature_vector)[0].max()
#         confidence_gb = model_gb.predict_proba(feature_vector)[0].max()
#     except Exception as e:
#         logging.error(f"Prediction error: {e}")
#         return jsonify({"error": "Prediction failed."}), 500

#     # Combine predictions (simple majority voting)
#     final_prediction = (prediction_rf + prediction_gb) // 2

#     # Prepare response
#     response = {
#         "final_prediction": "bot" if final_prediction == 1 else "human",
#         "model_confidences": {
#             "random_forest": confidence_rf,
#             "gradient_boosting": confidence_gb
#         },
#         "individual_predictions": {
#             "random_forest": "bot" if prediction_rf == 1 else "human",
#             "gradient_boosting": "bot" if prediction_gb == 1 else "human"
#         },
#         "extracted_features": extracted_features
#     }

#     return jsonify(response)

# # -------------------------------
# # API Route: Home
# # -------------------------------
# @app.route('/')
# def home():
#     """Serve the homepage."""
#     return send_from_directory('.', 'index.html')

# # -------------------------------
# # Main
# # -------------------------------
# if __name__ == '__main__':
#     try:
#         logging.info("Starting app.py...")
#         app.run(debug=True)
#     except KeyboardInterrupt:
#         logging.info("Shutting down app.py...")

from flask import Flask, request, jsonify
import logging
import pandas as pd
import joblib
from datetime import datetime
from data_feature_augmentation import extract_mouse_features, extract_key_features, extract_click_features

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = "models/bot_human_classifier.pkl"
FEATURES_PATH = "models/model_features.txt"

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        required_features = [line.strip() for line in f.readlines()]
    logging.info("✅ Model and feature list loaded successfully.")
except FileNotFoundError:
    logging.warning("⚠️ Model or feature list not found. Running in data collection mode only.")
    model = None
    required_features = []

def extract_features(data):
    """Extract combined features from user interaction data."""
    try:
        mouse_events = data.get("mouseEvents", [])
        key_events = data.get("keyEvents", [])
        click_events = data.get("clickEvents", [])

        # Extract features using specialized functions
        features = {}
        features.update(extract_mouse_features(mouse_events))
        features.update(extract_key_features(key_events))
        features.update(extract_click_features(click_events))

        return features
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        return {}

def validate_timing(data):
    """Validate timing data for human-like interaction."""
    try:
        mouse_events = pd.DataFrame(data.get("mouseEvents", []))
        key_events = pd.DataFrame(data.get("keyEvents", []))

        # Validate mouse movement speed
        if len(mouse_events) > 1:
            mouse_events["distance"] = ((mouse_events["x"].diff() ** 2 + mouse_events["y"].diff() ** 2) ** 0.5)
            mouse_events["time_diff"] = mouse_events["timestamp"].diff()
            mouse_events["speed"] = mouse_events["distance"] / mouse_events["time_diff"]
            avg_mouse_speed = mouse_events["speed"].mean()
            if avg_mouse_speed > 2.0:  # Adjust threshold as needed
                return False

        # Validate typing speed
        if len(key_events) > 1:
            key_events["time_diff"] = key_events["timestamp"].diff()
            avg_typing_speed = key_events["time_diff"].mean()
            if avg_typing_speed < 50:  # Too fast, likely a bot
                return False

        return True
    except Exception as e:
        logging.error(f"Error validating timing: {e}")
        return False

@app.route('/api/captcha-data', methods=['POST'])
def collect_data():
    data = request.json

    # Add metadata
    data['metadata'] = {
        'ip': request.remote_addr,
        'user_agent': request.headers.get('User-Agent'),
        'timestamp': datetime.now().isoformat()
    }

    # Validate timing
    is_human_like = validate_timing(data)

    # Make prediction if model is loaded
    if model:
        try:
            features = extract_features(data)
            features_df = pd.DataFrame([features])

            # Ensure feature vector matches model requirements
            for feature in required_features:
                if feature not in features_df:
                    features_df[feature] = 0

            features_df = features_df[required_features]
            proba = model.predict_proba(features_df)[0, 1]
            prediction = 1 if proba > 0.5 else 0

            result = {
                "status": "success",
                "is_bot": not is_human_like or bool(prediction),
                "confidence": float(proba)
            }
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            result = {"status": "error", "message": "Prediction failed"}
    else:
        result = {"status": "success", "message": "Running in data collection mode"}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)