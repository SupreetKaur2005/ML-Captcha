import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
data = pd.read_csv('data/processed/extracted_features.csv')

# Handle missing values
missing_values = data.isnull().sum()
if missing_values.sum() > 0:
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())

# Encode target variable if needed
if data['label'].dtype == 'O':
    data['label'] = data['label'].map({'human': 0, 'bot': 1})

# Separate features and target
X = data.drop('label', axis=1)
y = data['label']

# Correlation matrix
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("Correlation matrix saved as 'correlation_matrix.png'")
plt.close()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Check for class imbalance
train_class_counts = pd.Series(y_train).value_counts()
imbalance_ratio = min(train_class_counts) / max(train_class_counts)

# Create a pipeline with preprocessing and classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameters for grid search
reduced_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 20],
    'classifier__min_samples_split': [2]
}

grid_search = GridSearchCV(
    pipeline, 
    reduced_param_grid, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1,
    verbose=1
)

# Apply SMOTE if there's a significant class imbalance
if imbalance_ratio < 0.3:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    grid_search.fit(X_train_resampled, y_train_resampled)
else:
    grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Cross-validation predictions
cv_predictions = cross_val_predict(best_model, X, y, cv=5)

# Classification report after cross-validation
print("\nClassification Report (Cross-Validation):")
print(classification_report(y, cv_predictions, target_names=['human', 'bot']))

# Save the trained model
model_filename = 'human_bot_classifier.pkl'
joblib.dump(best_model, model_filename)
print(f"\nModel saved as '{model_filename}'")

# Feature importance
feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = X.columns

# Sort feature importances in descending order
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_importance = feature_importance[sorted_indices]
sorted_features = [feature_names[i] for i in sorted_indices]

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.bar(range(len(sorted_features)), sorted_importance, align='center')
plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")
plt.close()

# Example of how to use the model for prediction
def predict_human_or_bot(input_data, model=best_model):
    """
    Predict whether input data represents a human or bot.

    Parameters:
    input_data (dict or pandas DataFrame): User interaction data
    model: Trained classifier model

    Returns:
    str: 'human' or 'bot' prediction
    float: Probability of being a bot
    """
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    required_features = X.columns
    for feature in required_features:
        if feature not in input_data.columns:
            raise ValueError(f"Input data is missing required feature: {feature}")
    input_data = input_data[required_features]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    result = "bot" if prediction == 1 else "human"
    return result, probability

# Load model function for later use
def load_model(filename='human_bot_classifier.pkl'):
    """Load a trained model from a file."""
    return joblib.load(filename)

# Demonstrate prediction with a sample from the test set
sample_idx = 0
sample_input = X_test.iloc[sample_idx]
true_label = y_test.iloc[sample_idx]
prediction, probability = predict_human_or_bot(pd.DataFrame([sample_input]), best_model)