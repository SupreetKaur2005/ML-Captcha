# ML CAPTCHA: Machine Learning CAPTCHA for Real-time Behavior Assessment

**By:** Supreet Kaur and Ishita Garg

## Abstract

This project introduces a machine learning-based CAPTCHA system designed for real-time differentiation between human and bot interactions. By leveraging behavioral features such as mouse movements, typing dynamics, and click events, the system enhances security against sophisticated AI-driven CAPTCHA solvers. It employs feature engineering and a lightweight Random Forest Classifier to achieve accurate and efficient classification, ensuring usability and robustness for online platforms.

---

## 1. Introduction

Traditional CAPTCHA systems are becoming increasingly vulnerable to advanced AI solvers. ML CAPTCHA addresses this challenge by analyzing real-time behavioral patterns during user interactions. Through machine learning techniques, the system identifies nuanced differences between human and bot behaviors, providing a more robust and adaptive security mechanism. This report details the project’s objectives, data, methodology, and results.

---

## 2. Objective

The project aims to create a real-time CAPTCHA system powered by machine learning, with the following objectives:

- Extract behavioral patterns from user interactions (mouse, typing, clicks).
- Engineer interpretable features to differentiate between humans and bots.
- Train a lightweight machine learning model for real-time classification.
- Evaluate model performance using rigorous metrics and visualizations.
- Develop a deployment strategy for seamless integration into web applications.

---

## 3. Data Information

The project utilized interaction data from humans and bots. Key data categories include:

- **Mouse Interaction Data:** Features derived include acceleration variance, mean speed, and path curvature.
- **Typing Dynamics Data:** Features include keystroke intervals and backspace frequency.
- **Click Event Data:** Features such as spatial distribution and temporal intervals between clicks.
- **Session Metadata:** Includes interaction session durations.

Data was captured client-side using JavaScript and stored as JSON files. The `data_processing.py` script validated, structured, and labeled the data into `dataset.csv`. The `data_feature_augmentation.py` script extracted advanced features, saving them in `extracted_features.csv` for model training.

---

## 4. Motivation

The increasing sophistication of AI bots necessitates intelligent CAPTCHA systems. This project focuses on analyzing real-time behavioral patterns to create a more secure and user-friendly solution. The emphasis on feature interpretability and lightweight models ensures transparency and real-time efficiency, critical for deployment in web applications.

---

## 5. Methodology

The project pipeline spans data processing, feature engineering, model training, and real-time classification. Key details:

### **Data Processing**
- Script: `data_processing.py`
- Validates raw JSON data for required fields (`mouseEvents`, `keyEvents`, `clickEvents`).
- Labels interactions as human or bot, deduplicates records, and outputs `dataset.csv`.

### **Feature Engineering**
- Script: `data_feature_augmentation.py`
- Extracts advanced features:
  - **Mouse Events:** Speed, acceleration, curvature, duration.
  - **Typing Events:** Keystroke intervals, unique key count.
  - **Click Events:** Count, average interval, standard deviation.
- Outputs processed features to `extracted_features.csv`.

### **Model Training**
- Script: `train_model.py`
- Preprocessing:
  - Handles missing values and scales features using `StandardScaler`.
- Training:
  - Uses `RandomForestClassifier`.
  - Hyperparameter tuning via `GridSearchCV`.
  - Addresses class imbalance using SMOTE.
- Evaluation:
  - Stratified k-fold cross-validation.
  - Classification report and performance metrics.
- Outputs:
  - Trained model saved as `human_bot_classifier.pkl`.
  - Visualization of feature correlations (`correlation_matrix.png`).
  - Feature importance plot (`feature_importance.png`).

### **Real-time Classification**
- Script: `app.py`
- Flask API:
  - `/api/captcha-data`: Accepts interaction data, extracts features, and predicts using `human_bot_classifier.pkl`.
  - `/api/model-status`: Returns model status and required features.
- Classifies interactions in real-time as human or bot.

---

## 6. Result

### **Model Performance**
The Random Forest Classifier achieved high performance:

- **Precision:**
  - Humans: 96%
  - Bots: 100%
- **Recall:**
  - Humans: 100%
  - Bots: 96%
- **F1-Score:** 98% for both humans and bots.
- **Overall Accuracy:** 98%.

### **Insights**
- **Correlation Matrix:** Highlights relationships between features (`correlation_matrix.png`).
- **Feature Importance:** Identifies key features influencing classification decisions (`feature_importance.png`).

### **Deployment Readiness**
The trained model (`human_bot_classifier.pkl`) integrates seamlessly with the Flask API for real-time classification. These results validate the system’s robustness and accuracy in distinguishing between human and bot behaviors.

---

## Conclusion

The ML CAPTCHA system effectively differentiates between human and bot interactions using real-time behavioral data. The combination of interpretable feature engineering and a lightweight Random Forest classifier ensures high classification performance while maintaining efficiency. The system is secure, user-friendly, and ready for deployment in web applications.

---

## Future Work

- **Advanced Feature Engineering:** Explore deep behavioral patterns for enhanced classification.
- **Model Optimization:** Evaluate other lightweight machine learning models.
- **Adversarial Testing:** Test against more sophisticated bot simulations.
- **Adaptive Mechanisms:** Develop dynamic systems that evolve with changing bot behaviors.