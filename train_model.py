# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# # Step 1: Define paths using environment variables or default values
# DATA_PATH = os.getenv("DATA_PATH", "data/processed/extracted_features.csv")
# MODEL_DIR = os.getenv("MODEL_DIR", "models")
# RF_MODEL_PATH = os.path.join(MODEL_DIR, "optimized_rf_classifier.pkl")
# GB_MODEL_PATH = os.path.join(MODEL_DIR, "gradient_boosting_classifier.pkl")
# FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.txt")

# # Ensure the model directory exists
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Step 2: Load the extracted feature dataset
# try:
#     df = pd.read_csv(DATA_PATH)
# except FileNotFoundError:
#     raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please verify the path.")

# # Step 3: Prepare features and label
# X = df.drop(columns=["label"])
# y = df["label"]

# # Step 4: Encode the target label ('human', 'bot') into integers
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)  # human=0, bot=1 (or vice versa)

# # Step 5: Feature scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Step 6: Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )

# # Step 7: Hyperparameter tuning using GridSearchCV
# param_grid = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
# }
# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_grid=param_grid,
#     cv=3,
#     n_jobs=-1,
#     verbose=2,
# )
# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_

# # Step 8: Train a Gradient Boosting model (optional enhancement)
# gb_model = GradientBoostingClassifier(random_state=42)
# gb_model.fit(X_train, y_train)

# # Step 9: Save the models and feature list
# joblib.dump(best_model, RF_MODEL_PATH)
# joblib.dump(gb_model, GB_MODEL_PATH)
# with open(FEATURES_PATH, "w") as f:
#     for feature in X.columns:
#         f.write(f"{feature}\n")

# print(f"✅ Models and feature list saved to {MODEL_DIR}.")

# # Step 10: Make predictions and evaluate
# y_pred_rf = best_model.predict(X_test)
# y_pred_gb = gb_model.predict(X_test)

# print("Random Forest Classifier Evaluation:")
# print("Accuracy:", accuracy_score(y_test, y_pred_rf))
# print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

# print("\nGradient Boosting Classifier Evaluation:")
# print("Accuracy:", accuracy_score(y_test, y_pred_gb))
# print("\nClassification Report:\n", classification_report(y_test, y_pred_gb, target_names=label_encoder.classes_))

# # Step 11: Confusion Matrix for both models
# cm_rf = confusion_matrix(y_test, y_pred_rf)
# cm_gb = confusion_matrix(y_test, y_pred_gb)

# # Plot for Random Forest
# sns.heatmap(cm_rf, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix - Random Forest")
# plt.show()

# # Plot for Gradient Boosting
# sns.heatmap(cm_gb, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Greens")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix - Gradient Boosting")
# plt.show()


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Define paths using environment variables or default values
DATA_PATH = os.getenv("DATA_PATH", "data/processed/extracted_features.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "optimized_rf_classifier.pkl")
GB_MODEL_PATH = os.path.join(MODEL_DIR, "gradient_boosting_classifier.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.txt")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Step 2: Load the dataset
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please verify the path.")

# Step 3: Feature Engineering
# Add new features to enhance the model
df["mouse_path_irregularity"] = df["mouse_avg_curvature"] / df["mouse_total_distance"]
df["click_intensity"] = df["click_event_count"] / df["mouse_duration_ms"]
df["keystroke_efficiency"] = df["key_unique_count"] / df["key_event_count"]
df["mouse_efficiency"] = df["mouse_total_distance"] / df["mouse_duration_ms"]

# Replace infinite or NaN values with 0 (e.g., division by zero handling)
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Step 4: Prepare features and label
X = df.drop(columns=["label"])
y = df["label"]

# Step 5: Encode the target label ('human', 'bot') into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # human=0, bot=1 (or vice versa)

# Step 6: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Step 8: Hyperparameter tuning using GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Step 9: Train a Gradient Boosting model (optional enhancement)
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Step 10: Save the models and feature list
joblib.dump(best_model, RF_MODEL_PATH)
joblib.dump(gb_model, GB_MODEL_PATH)
with open(FEATURES_PATH, "w") as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

print(f"✅ Models and feature list saved to {MODEL_DIR}.")

# Step 11: Make predictions and evaluate
y_pred_rf = best_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

print("Random Forest Classifier Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

print("\nGradient Boosting Classifier Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gb, target_names=label_encoder.classes_))


# Step 12: Confusion Matrix for both models with visualization
cm_rf = confusion_matrix(y_test, y_pred_rf)  # Define cm_rf
cm_gb = confusion_matrix(y_test, y_pred_gb)  # Ensure cm_gb is also defined

def plot_confusion_matrix(cm, model_name, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

plot_confusion_matrix(cm_rf, "Random Forest Classifier", label_encoder.classes_)
plot_confusion_matrix(cm_gb, "Gradient Boosting Classifier", label_encoder.classes_)

# Step 13: Feature Importance Visualization for Random Forest
def plot_feature_importance(importance, features, model_name, top_n=10):
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title(f"Top {top_n} Important Features - {model_name}")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.show()

# Feature importance for Random Forest
if hasattr(best_model, "feature_importances_"):
    plot_feature_importance(best_model.feature_importances_, X.columns, "Random Forest")

# Feature importance for Gradient Boosting
if hasattr(gb_model, "feature_importances_"):
    plot_feature_importance(gb_model.feature_importances_, X.columns, "Gradient Boosting")

# Step 14: Accuracy Comparison Visualization
models = ["Random Forest", "Gradient Boosting"]
accuracies = [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_gb)]

plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=accuracies, palette="coolwarm")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()