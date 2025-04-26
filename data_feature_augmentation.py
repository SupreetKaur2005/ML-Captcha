import pandas as pd
import numpy as np
import os
import ast
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p2) - np.array(p1))


def extract_mouse_features(mouse_events):
    if not mouse_events or len(mouse_events) < 2:
        return {
            "mouse_movement_count": 0,
            "mouse_total_distance": 0.0,
            "mouse_avg_speed": 0.0,
            "mouse_std_speed": 0.0,
            "mouse_max_speed": 0.0,
            "mouse_avg_acceleration": 0.0,
            "mouse_std_acceleration": 0.0,
            "mouse_avg_curvature": 0.0,
            "mouse_duration_ms": 0,
        }

    positions = [(e["x"], e["y"]) for e in mouse_events]
    timestamps = [e["timestamp"] for e in mouse_events]

    distances = [compute_distance(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]
    time_deltas = [(timestamps[i + 1] - timestamps[i]) for i in range(len(timestamps) - 1)]

    speeds = [d / t if t > 0 else 0 for d, t in zip(distances, time_deltas)]
    accelerations = [speeds[i + 1] - speeds[i] for i in range(len(speeds) - 1)]
    curvatures = []
    for i in range(1, len(positions) - 1):
        a, b, c = positions[i - 1], positions[i], positions[i + 1]
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        curvatures.append(angle)

    return {
        "mouse_movement_count": len(mouse_events),
        "mouse_total_distance": sum(distances),
        "mouse_avg_speed": np.mean(speeds) if speeds else 0,
        "mouse_std_speed": np.std(speeds) if speeds else 0,
        "mouse_max_speed": max(speeds) if speeds else 0,
        "mouse_avg_acceleration": np.mean(accelerations) if accelerations else 0,
        "mouse_std_acceleration": np.std(accelerations) if accelerations else 0,
        "mouse_avg_curvature": np.mean(curvatures) if curvatures else 0,
        "mouse_duration_ms": timestamps[-1] - timestamps[0] if timestamps else 0,
    }


def extract_key_features(key_events):
    if not key_events:
        return {
            "key_event_count": 0,
            "key_unique_count": 0,
            "key_avg_interval": 0.0,
            "key_std_interval": 0.0,
        }

    try:
        timestamps = [e["timestamp"] for e in key_events if "timestamp" in e]
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        key_counts = len(key_events)
        unique_keys = len(set(e.get("key", "UNKNOWN") for e in key_events))  # Handle missing keys

        return {
            "key_event_count": key_counts,
            "key_unique_count": unique_keys,
            "key_avg_interval": np.mean(intervals) if intervals else 0,
            "key_std_interval": np.std(intervals) if intervals else 0,
        }
    except Exception as e:
        logging.warning(f"Error processing key events: {e}")
        return {
            "key_event_count": 0,
            "key_unique_count": 0,
            "key_avg_interval": 0.0,
            "key_std_interval": 0.0,
        }


def extract_click_features(click_events):
    if not click_events or len(click_events) < 2:
        return {
            "click_event_count": 0,
            "click_avg_interval": 0.0,
            "click_std_interval": 0.0,
        }

    timestamps = [e["timestamp"] for e in click_events if "timestamp" in e]
    intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

    return {
        "click_event_count": len(click_events),
        "click_avg_interval": np.mean(intervals) if intervals else 0,
        "click_std_interval": np.std(intervals) if intervals else 0,
    }


# Load the dataset
input_path = os.path.join("data", "processed", "dataset.csv")
output_path = os.path.join("data", "processed", "extracted_features.csv")

try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    logging.error(f"Dataset not found at {input_path}. Please ensure the file exists.")
    exit()

# Feature extraction loop
features = []
for _, row in df.iterrows():
    try:
        mouse_data = ast.literal_eval(row.get("mouseEvents", "[]"))
        key_data = ast.literal_eval(row.get("keyEvents", "[]"))
        click_data = ast.literal_eval(row.get("clickEvents", "[]"))

        feature_row = {}
        feature_row.update(extract_mouse_features(mouse_data))
        feature_row.update(extract_key_features(key_data))
        feature_row.update(extract_click_features(click_data))
        feature_row["label"] = row["label"]

        features.append(feature_row)
    except (ValueError, SyntaxError, KeyError) as e:
        logging.warning(f"Skipping row due to error: {e}")

# Save the extracted features
df_features = pd.DataFrame(features)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_features.to_csv(output_path, index=False)
logging.info(f"Extracted features saved to {output_path}.")