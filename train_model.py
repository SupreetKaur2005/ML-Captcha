# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import pandas as pd

# # Load the dataset
# file_path = "data/processed/augmented_dataset.csv"
# data = pd.read_csv(file_path)

# # Define the features and target variable
# X = data.drop(columns=['label'])
# y = data['label']

# # Define the column transformer for preprocessing
# numeric_features = ['movement_count', 'total_distance', 'avg_speed', 'std_speed', 'max_speed', 'avg_acceleration', 'std_acceleration', 'avg_curvature', 'duration_ms']
# categorical_features = ['keyEvents', 'clickEvents', 'scrollEvents', 'inputEvents', 'sessionInfo', 'metadata']

# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)])

# # Encode the target variable
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# # Define the model pipeline
# model = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('classifier', RandomForestClassifier(random_state=42))])

# # Train the model
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# # Example new data point
# new_data = pd.DataFrame({
#     'movement_count': [1],
#     'total_distance': [100.5],
#     'avg_speed': [5.2],
#     'std_speed': [1.3],
#     'max_speed': [10],
#     'avg_acceleration': [2],
#     'std_acceleration': [1],
#     'avg_curvature': [0.5],
#     'duration_ms': [5000],
#     'keyEvents': ['event1'],
#     'clickEvents': ['click1'],
#     'scrollEvents': ['scroll1'],
#     'inputEvents': ['input1'],
#     'sessionInfo': ['info1'],
#     'metadata': ['meta1']
# })

# # Make a prediction
# prediction = model.predict(new_data)
# predicted_label = label_encoder.inverse_transform(prediction)
# print("Predicted Label:", predicted_label[0])


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_and_train(data):
    try:
        # Define the features and target variable
        X = data.drop(columns=['label'])
        y = data['label']

        # Define the column transformer for preprocessing
        numeric_features = ['movement_count', 'total_distance', 'avg_speed', 'std_speed', 'max_speed', 'avg_acceleration', 'std_acceleration', 'avg_curvature', 'duration_ms']
        categorical_features = ['keyEvents', 'clickEvents', 'scrollEvents', 'inputEvents', 'sessionInfo', 'metadata']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Encode the target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Define the model pipeline
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(random_state=42))])

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{report}")

        # Save the model and feature names
        joblib.dump(model, 'captcha_model.pkl')
        with open('model_features.txt', 'w') as f:
            f.write("\n".join(numeric_features + categorical_features))

        return model, label_encoder
    except Exception as e:
        logging.error(f"Error in preprocessing and training: {e}")
        sys.exit(1)

def extract_features(data):
    """Extract the same features used during training from real-time data"""
    features = {}

    # Mouse dynamics
    mouse_events = data.get('mouseEvents', [])
    if len(mouse_events) >= 3:
        df = pd.DataFrame(mouse_events)
        df['timestamp_diff'] = df['timestamp'].diff().fillna(0)
        df['distance'] = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2).fillna(0)
        df['speed'] = np.where(df['timestamp_diff'] > 0, df['distance'] / df['timestamp_diff'], 0)
        df['acceleration'] = df['speed'].diff().fillna(0)

        # Calculate curvature
        curvatures = []
        for i in range(1, len(df) - 1):
            if i > 0 and i < len(df) - 1:
                dx1, dy1 = df.iloc[i]['x'] - df.iloc[i-1]['x'], df.iloc[i]['y'] - df.iloc[i-1]['y']
                dx2, dy2 = df.iloc[i+1]['x'] - df.iloc[i]['x'], df.iloc[i+1]['y'] - df.iloc[i]['y']

                if dx1 == 0 or dx2 == 0:
                    curvatures.append(0)
                    continue

                angle1 = np.arctan(dy1 / dx1)
                angle2 = np.arctan(dy2 / dx2)
                curvature = abs(angle2 - angle1)
                curvatures.append(curvature)

        features.update({
            'mouse_count': len(df),
            'mouse_mean_speed': df['speed'].mean(),
            'mouse_std_speed': df['speed'].std(),
            'mouse_max_speed': df['speed'].max(),
            'mouse_acceleration_var': df['acceleration'].var(),
            'mouse_path_length': df['distance'].sum(),
            'mouse_mean_curvature': np.mean(curvatures) if curvatures else 0,
            'mouse_std_curvature': np.std(curvatures) if curvatures else 0,
            'mouse_x_std': df['x'].std(),
            'mouse_y_std': df['y'].std(),
        })

    # Keyboard dynamics
    key_events = data.get('keyEvents', [])
    if len(key_events) >= 2:
        df = pd.DataFrame(key_events)

        keystroke_intervals = []
        backspace_count = 0
        error_indicators = 0

        for i in range(1, len(df)):
            if df.iloc[i-1]['eventType'] == 'keydown' and df.iloc[i]['eventType'] == 'keyup' and df.iloc[i-1]['key'] == df.iloc[i]['key']:
                interval = df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']
                keystroke_intervals.append(interval)

            if df.iloc[i]['key'] == 'Backspace' and df.iloc[i]['eventType'] == 'keydown':
                backspace_count += 1
                error_indicators += 1

        features.update({
            'key_event_count': len(df),
            'unique_keys': len(df['key'].unique()),
            'backspace_frequency': backspace_count / len(df) if len(df) > 0 else 0,
            'error_rate': error_indicators / len(df) if len(df) > 0 else 0,
            'keystroke_mean_interval': np.mean(keystroke_intervals) if keystroke_intervals else 0,
            'keystroke_std_interval': np.std(keystroke_intervals) if keystroke_intervals else 0,
            'keystroke_max_interval': np.max(keystroke_intervals) if keystroke_intervals else 0,
            'keystroke_min_interval': np.min(keystroke_intervals) if keystroke_intervals else 0,
        })

    # Click patterns
    click_events = data.get('clickEvents', [])
    if len(click_events) >= 2:
        df = pd.DataFrame(click_events)
        df['time_between_clicks'] = df['timestamp'].diff().fillna(0)

        features.update({
            'click_count': len(df),
            'click_x_std': df['x'].std(),
            'click_y_std': df['y'].std(),
            'click_mean_interval': df['time_between_clicks'].mean(),
            'click_std_interval': df['time_between_clicks'].std(),
        })

    # Scroll behavior
    scroll_events = data.get('scrollEvents', [])
    if len(scroll_events) >= 3:
        df = pd.DataFrame(scroll_events)
        df['timestamp_diff'] = df['timestamp'].diff().fillna(0)
        df['scroll_diff'] = df['scrollY'].diff().fillna(0)

        df['scroll_speed'] = np.where(df['timestamp_diff'] > 0, df['scroll_diff'] / df['timestamp_diff'], 0)
        df['scroll_acceleration'] = df['scroll_speed'].diff().fillna(0)
        df['scroll_jerk'] = df['scroll_acceleration'].diff().fillna(0)

        features.update({
            'scroll_count': len(df),
            'scroll_mean_speed': df['scroll_speed'].mean(),
            'scroll_std_speed': df['scroll_speed'].std(),
            'scroll_mean_acceleration': df['scroll_acceleration'].mean(),
            'scroll_std_acceleration': df['scroll_acceleration'].std(),
            'scroll_jerkiness': df['scroll_jerk'].abs().mean(),
        })

    # Session features
    session_info = data.get('sessionInfo', {})
    duration = session_info.get('duration', 0)

    total_events = (
        len(data.get('mouseEvents', [])) +
        len(data.get('keyEvents', [])) +
        len(data.get('clickEvents', [])) +
        len(data.get('scrollEvents', []))
    )

    features.update({
        'session_duration': duration,
        'total_event_count': total_events,
        'events_per_second': total_events / (duration / 1000) if duration > 0 else 0,
    })

    return features

def predict(model, label_encoder, new_data):
    try:
        features = extract_features(new_data)
        features_df = pd.DataFrame([features])

        # Ensure all required features exist
        with open('model_features.txt', 'r') as f:
            required_features = [line.strip() for line in f.readlines()]

        for feature in required_features:
            if feature not in features_df:
                features_df[feature] = 0

        features_df = features_df[required_features]

        prediction = model.predict(features_df)
        predicted_label = label_encoder.inverse_transform(prediction)
        logging.info(f"Predicted Label: {predicted_label[0]}")
        return predicted_label[0]
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        sys.exit(1)

def main(file_path, new_data):
    data = load_data(file_path)
    model, label_encoder = preprocess_and_train(data)
    prediction = predict(model, label_encoder, new_data)
    return prediction

if __name__ == "__main__":
    file_path = 'data/processed/augmented_dataset.csv'
    new_data = {
        'mouseEvents': [],
        'keyEvents': [],
        'clickEvents': [],
        'scrollEvents': [],
        'sessionInfo': {'duration': 5000},
        'metadata': {}
    }
    prediction = main(file_path, new_data)
    print(f"Predicted Label: {prediction}")
