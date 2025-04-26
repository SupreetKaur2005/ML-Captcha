from flask import Flask, request, jsonify, send_from_directory
import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the trained model
try:
    model = joblib.load('human_bot_classifier.pkl')
    # Get the feature names from the model
    # Assuming the model is a pipeline with a classifier as the last step
    # and the classifier has feature_names_in_ attribute (scikit-learn >= 1.0)
    logging.info("✅ Model loaded successfully.")
except FileNotFoundError:
    logging.error("❌ Model 'human_bot_classifier.pkl' not found!")
    model = None

# Get required features based on your output
required_features = [
    'mouse_movement_count', 'mouse_total_distance', 'mouse_avg_speed',
    'mouse_std_speed', 'mouse_max_speed', 'mouse_avg_acceleration',
    'mouse_std_acceleration', 'mouse_avg_curvature', 'mouse_duration_ms',
    'key_event_count', 'key_unique_count', 'key_avg_interval',
    'key_std_interval', 'click_event_count', 'click_avg_interval',
    'click_std_interval'
]

# Feature Extraction Function
def extract_features(data):
    """Extract features from real-time data."""
    # Extract mouse movement features
    mouse_events = data.get("mouseEvents", [])
    key_events = data.get("keyEvents", [])
    click_events = data.get("clickEvents", [])
    
    features = {}
    
    # Process mouse events
    if mouse_events and len(mouse_events) >= 2:
        # Extract positions and timestamps
        positions = [(e["x"], e["y"]) for e in mouse_events]
        timestamps = [e["timestamp"] for e in mouse_events]
        
        # Calculate distances between consecutive points
        distances = [np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i])) 
                    for i in range(len(positions) - 1)]
        
        # Calculate time deltas
        deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)]
        
        # Calculate speeds (distance/time)
        speeds = [d / max(t, 1) for d, t in zip(distances, deltas)]  # Avoid division by zero
        
        # Calculate accelerations (speed changes)
        accelerations = [speeds[i+1] - speeds[i] for i in range(len(speeds) - 1)] if len(speeds) > 1 else [0]
        
        # Calculate curvature
        # For simplicity, using angle changes between consecutive segments
        curvatures = []
        for i in range(len(positions) - 2):
            v1 = np.array(positions[i+1]) - np.array(positions[i])
            v2 = np.array(positions[i+2]) - np.array(positions[i+1])
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                # Calculate dot product and angle
                dot_product = np.dot(v1, v2) / (v1_norm * v2_norm)
                # Clip to handle floating point errors
                dot_product = max(min(dot_product, 1.0), -1.0)
                angle = np.arccos(dot_product)
                curvatures.append(angle)
            else:
                curvatures.append(0)
        
        # Populate mouse movement features
        features['mouse_movement_count'] = len(mouse_events)
        features['mouse_total_distance'] = sum(distances)
        features['mouse_avg_speed'] = np.mean(speeds) if speeds else 0
        features['mouse_std_speed'] = np.std(speeds) if speeds else 0
        features['mouse_max_speed'] = max(speeds) if speeds else 0
        features['mouse_avg_acceleration'] = np.mean(accelerations) if accelerations else 0
        features['mouse_std_acceleration'] = np.std(accelerations) if accelerations else 0
        features['mouse_avg_curvature'] = np.mean(curvatures) if curvatures else 0
        features['mouse_duration_ms'] = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    else:
        # No mouse events or not enough data
        features['mouse_movement_count'] = 0
        features['mouse_total_distance'] = 0
        features['mouse_avg_speed'] = 0
        features['mouse_std_speed'] = 0
        features['mouse_max_speed'] = 0
        features['mouse_avg_acceleration'] = 0
        features['mouse_std_acceleration'] = 0
        features['mouse_avg_curvature'] = 0
        features['mouse_duration_ms'] = 0
    
    # Process keyboard events
    if key_events and len(key_events) >= 2:
        key_timestamps = [e["timestamp"] for e in key_events]
        key_codes = [e.get("keyCode", e.get("code", "")) for e in key_events]
        unique_keys = set(key_codes)
        
        # Calculate intervals between key presses
        key_intervals = [key_timestamps[i+1] - key_timestamps[i] for i in range(len(key_timestamps) - 1)]
        
        features['key_event_count'] = len(key_events)
        features['key_unique_count'] = len(unique_keys)
        features['key_avg_interval'] = np.mean(key_intervals) if key_intervals else 0
        features['key_std_interval'] = np.std(key_intervals) if len(key_intervals) > 1 else 0
    else:
        features['key_event_count'] = 0
        features['key_unique_count'] = 0
        features['key_avg_interval'] = 0
        features['key_std_interval'] = 0
    
    # Process click events
    if click_events and len(click_events) >= 2:
        click_timestamps = [e["timestamp"] for e in click_events]
        
        # Calculate intervals between clicks
        click_intervals = [click_timestamps[i+1] - click_timestamps[i] for i in range(len(click_timestamps) - 1)]
        
        features['click_event_count'] = len(click_events)
        features['click_avg_interval'] = np.mean(click_intervals) if click_intervals else 0
        features['click_std_interval'] = np.std(click_intervals) if len(click_intervals) > 1 else 0
    else:
        features['click_event_count'] = 0
        features['click_avg_interval'] = 0
        features['click_std_interval'] = 0
    
    return features

# API Route: Collect & Predict
@app.route('/api/captcha-data', methods=['POST'])
def collect_data():
    try:
        data = request.json
        data['metadata'] = {
            'ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save data to the `data/raw` directory
        save_directory = 'data/raw'
        os.makedirs(save_directory, exist_ok=True)
        filename = f"{save_directory}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.remote_addr}.json"
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        logging.info(f"Received and saved data to {filename}")
        
        # Make prediction if model is loaded
        if model:
            try:
                # Extract features based on the events data
                features = extract_features(data)
                
                # Create DataFrame with the extracted features
                features_df = pd.DataFrame([features])
                
                # Ensure all required features are present
                for feature in required_features:
                    if feature not in features_df:
                        features_df[feature] = 0
                
                # Ensure correct feature order
                features_df = features_df[required_features]
                
                # Make prediction using the loaded model
                # The model pipeline already includes scaling
                prediction = model.predict(features_df)[0]
                probability = model.predict_proba(features_df)[0][1]  # Probability of being a bot
                
                # Log the prediction details
                logging.info(f"Prediction: {'bot' if prediction == 1 else 'human'} with probability {probability:.4f}")
                
                result = {
                    "status": "success",
                    "is_bot": bool(prediction == 1),  # True if bot, False if human
                    "confidence": float(probability),
                    "challenge_required": bool(probability > 0.3)
                }
                
                # Save prediction results
                prediction_dir = 'data/predictions'
                os.makedirs(prediction_dir, exist_ok=True)
                pred_filename = f"{prediction_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.remote_addr}.json"
                with open(pred_filename, 'w') as f:
                    json.dump({
                        "features": features,
                        "prediction": "bot" if prediction == 1 else "human",
                        "probability": float(probability),
                        "timestamp": datetime.now().isoformat()
                    }, f)
                
            except Exception as e:
                logging.error(f"Prediction error: {str(e)}")
                result = {
                    "status": "error",
                    "challenge_required": True,
                    "message": f"Error making prediction: {str(e)}"
                }
        else:
            result = {
                "status": "warning",
                "challenge_required": True,
                "message": "Running in data collection mode (model not loaded)"
            }

        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Request processing error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500

# API Route: Home
@app.route('/')
def home():
    """Serve the homepage."""
    return send_from_directory('.', 'index.html')

# Endpoint to check model status
@app.route('/api/model-status', methods=['GET'])
def model_status():
    if model:
        return jsonify({
            "status": "active",
            "model_file": "human_bot_classifier.pkl",
            "features": required_features
        })
    else:
        return jsonify({
            "status": "inactive",
            "message": "Model not loaded"
        })

# Main
if __name__ == '__main__':
    try:
        logging.info("Starting CAPTCHA detection server...")
        logging.info(f"Model status: {'Loaded' if model else 'Not loaded'}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logging.info("Shutting down server...")