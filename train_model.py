# 

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# -------------------------------
# Step 1: Load the Dataset
# -------------------------------
print("📥 Loading dataset...")
df = pd.read_csv("data/processed/extracted_features.csv")

# Check for missing columns
required_columns = ["avg_speed", "total_distance", "std_acceleration", "avg_curvature", "std_curvature"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"❌ Missing required columns in the dataset: {missing_columns}")

if df.isnull().any().any():
    print("⚠️ Missing values detected. Filling with zeros...")
    df.fillna(0, inplace=True)

# -------------------------------
# Step 2: Feature Engineering
# -------------------------------
print("🛠️ Enhancing features...")
# Add interaction terms or derived features for better model performance
df["speed_to_distance_ratio"] = df["avg_speed"] / (df["total_distance"] + 1e-6)
df["acceleration_to_speed_ratio"] = df["std_acceleration"] / (df["avg_speed"] + 1e-6)
df["curvature_variability"] = df["avg_curvature"] / (df["std_curvature"] + 1e-6)

# Prepare features and labels
X = df.drop(columns=["label"])
y = df["label"]

# -------------------------------
# Step 3: Encode Labels
# -------------------------------
print("🔤 Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # human=0, bot=1 (or vice versa)

# -------------------------------
# Step 4: Train-Test Split
# -------------------------------
print("✂️ Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------------
# Step 5: Model Training with Hyperparameter Tuning
# -------------------------------
print("🚀 Training the model with hyperparameter tuning...")
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
print("✅ Best parameters found:", grid_search.best_params_)

# -------------------------------
# Step 6: Save the Model and Features
# -------------------------------
print("💾 Saving the trained model and feature list...")
joblib.dump(model, "bot_human_classifier.pkl")
with open("model_features.txt", "w") as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

# -------------------------------
# Step 7: Model Evaluation
# -------------------------------
print("📊 Evaluating the model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# Step 8: Feature Importances
# -------------------------------
print("📈 Plotting feature importances...")
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
plt.title("Feature Importances")
plt.show()

print("✅ Model training and evaluation completed.")