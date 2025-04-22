import pandas as pd
import numpy as np
import ast

def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p2) - np.array(p1))

def extract_mouse_features(mouse_events):
    if not mouse_events or len(mouse_events) < 2:
        return {
            "movement_count": 0,
            "total_distance": 0.0,
            "avg_speed": 0.0,
            "std_speed": 0.0,
            "max_speed": 0.0,
            "avg_acceleration": 0.0,
            "std_acceleration": 0.0,
            "avg_curvature": 0.0,
            "duration_ms": 0
        }

    positions = [(e["x"], e["y"]) for e in mouse_events]
    timestamps = [e["timestamp"] for e in mouse_events]

    distances = [compute_distance(positions[i], positions[i+1]) for i in range(len(positions)-1)]
    time_deltas = [(timestamps[i+1] - timestamps[i]) for i in range(len(timestamps)-1)]

    speeds = [d / t if t > 0 else 0 for d, t in zip(distances, time_deltas)]
    accelerations = [speeds[i+1] - speeds[i] for i in range(len(speeds)-1)]
    curvatures = []
    for i in range(1, len(positions)-1):
        a, b, c = positions[i-1], positions[i], positions[i+1]
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        curvatures.append(angle)

    return {
        "movement_count": len(mouse_events),
        "total_distance": sum(distances),
        "avg_speed": np.mean(speeds) if speeds else 0,
        "std_speed": np.std(speeds) if speeds else 0,
        "max_speed": max(speeds) if speeds else 0,
        "avg_acceleration": np.mean(accelerations) if accelerations else 0,
        "std_acceleration": np.std(accelerations) if accelerations else 0,
        "avg_curvature": np.mean(curvatures) if curvatures else 0,
        "duration_ms": timestamps[-1] - timestamps[0] if timestamps else 0
    }

# Load your dataset
df = pd.read_csv("data\processed\dataset.csv")

# Feature extraction loop
features = []
for _, row in df.iterrows():
    mouse_data = ast.literal_eval(row["mouseEvents"])
    feature_row = extract_mouse_features(mouse_data)
    feature_row["label"] = row["label"]
    features.append(feature_row)

df_features = pd.DataFrame(features)
df_features.to_csv("data\processed\extracted_features.csv", index=False)
