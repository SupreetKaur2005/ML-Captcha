# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Step 1: Load the extracted feature dataset
# df = pd.read_csv('data/processed/extracted_features.csv')

# # Step 2: Prepare features and label
# X = df.drop(columns=["label"])
# y = df["label"]

# # Step 3: Encode the target label ('human', 'bot') into integers
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)  # human=0, bot=1 (or vice versa)

# # Step 4: Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# # Step 5: Train Random Forest Classifier
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Step 6: Make predictions
# y_pred = model.predict(X_test)

# # Step 7: Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# # Step 8: Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Step 5: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Save the model and feature list
joblib.dump(model, "bot_human_classifier.pkl")
with open("model_features.txt", "w") as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

print("✅ Model and feature list saved.")

# Step 7: Make predictions and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print("✅ Model evaluation completed.")