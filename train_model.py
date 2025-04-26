import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Load the extracted feature dataset
df = pd.read_csv("data/processed/extracted_features.csv")

# Step 2: Prepare features and label
X = df.drop(columns=["label"])
y = df["label"]

# Step 3: Encode the target label ('human', 'bot') into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # human=0, bot=1 (or vice versa)

# Step 4: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Step 6: Hyperparameter tuning using GridSearchCV
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

# Step 7: Train a Gradient Boosting model (optional enhancement)
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Step 8: Save the models and feature list
joblib.dump(best_model, "optimized_rf_classifier.pkl")
joblib.dump(gb_model, "gradient_boosting_classifier.pkl")
with open("model_features.txt", "w") as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

print("✅ Models and feature list saved.")

# Step 9: Make predictions and evaluate
y_pred_rf = best_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

print("Random Forest Classifier Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

print("\nGradient Boosting Classifier Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gb, target_names=label_encoder.classes_))

# Step 10: Confusion Matrix for both models
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_gb = confusion_matrix(y_test, y_pred_gb)

# Plot for Random Forest
sns.heatmap(cm_rf, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Plot for Gradient Boosting
sns.heatmap(cm_gb, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Gradient Boosting")
plt.show()