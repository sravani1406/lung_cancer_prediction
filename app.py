import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the scaler and feature columns
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Load all models
models = {
    'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
    'K-Nearest Neighbors': joblib.load('k-nearest_neighbors_model.pkl'),
    'Decision Tree': joblib.load('decision_tree_model.pkl'),
    'Support Vector Machine': joblib.load('support_vector_machine_model.pkl'),
    'Naive Bayes': joblib.load('naive_bayes_model.pkl'),
    'Random Forest': joblib.load('random_forest_model.pkl')
}

# Streamlit app
st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")

# Title and description
st.title("Lung Cancer Prediction System")
st.markdown("""
This application predicts the likelihood of lung cancer based on various risk factors and symptoms.
Select a machine learning model, provide patient details in the sidebar, and click 'Predict' to see the result.
""")

# Sidebar for input
st.sidebar.header("Patient Details")

# Model selection
model_name = st.sidebar.selectbox("Select Machine Learning Model", list(models.keys()))

# Input fields
gender = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
smoking = st.sidebar.radio("Smoking", ["Yes", "No"])
yellow_fingers = st.sidebar.radio("Yellow Fingers", ["Yes", "No"])
anxiety = st.sidebar.radio("Anxiety", ["Yes", "No"])
peer_pressure = st.sidebar.radio("Peer Pressure", ["Yes", "No"])
chronic_disease = st.sidebar.radio("Chronic Disease", ["Yes", "No"])
fatigue = st.sidebar.radio("Fatigue", ["Yes", "No"])
allergy = st.sidebar.radio("Allergy", ["Yes", "No"])
wheezing = st.sidebar.radio("Wheezing", ["Yes", "No"])
alcohol_consuming = st.sidebar.radio("Alcohol Consuming", ["Yes", "No"])
coughing = st.sidebar.radio("Coughing", ["Yes", "No"])
shortness_of_breath = st.sidebar.radio("Shortness of Breath", ["Yes", "No"])
swallowing_difficulty = st.sidebar.radio("Swallowing Difficulty", ["Yes", "No"])
chest_pain = st.sidebar.radio("Chest Pain", ["Yes", "No"])

# Predict button
if st.sidebar.button("Predict"):
    # Prepare input data
    input_data = {
        'GENDER': 1 if gender == "Male" else 0,
        'AGE': age,
        'SMOKING': 2 if smoking == "Yes" else 1,
        'YELLOW_FINGERS': 2 if yellow_fingers == "Yes" else 1,
        'ANXIETY': 2 if anxiety == "Yes" else 1,
        'PEER_PRESSURE': 2 if peer_pressure == "Yes" else 1,
        'CHRONIC_DISEASE': 2 if chronic_disease == "Yes" else 1,
        'FATIGUE': 2 if fatigue == "Yes" else 1,
        'ALLERGY': 2 if allergy == "Yes" else 1,
        'WHEEZING': 2 if wheezing == "Yes" else 1,
        'ALCOHOL_CONSUMING': 2 if alcohol_consuming == "Yes" else 1,
        'COUGHING': 2 if coughing == "Yes" else 1,
        'SHORTNESS_OF_BREATH': 2 if shortness_of_breath == "Yes" else 1,
        'SWALLOWING_DIFFICULTY': 2 if swallowing_difficulty == "Yes" else 1,
        'CHEST_PAIN': 2 if chest_pain == "Yes" else 1
    }

    # Create DataFrame and ensure column order
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    
    # Scale the input if required
    if model_name in ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine']:
        input_scaled = scaler.transform(input_df)
        prediction = models[model_name].predict(input_scaled)[0]
        prediction_proba = models[model_name].predict_proba(input_scaled)[0]
    else:
        prediction = models[model_name].predict(input_df)[0]
        prediction_proba = models[model_name].predict_proba(input_df)[0]
    
    # Display result
    st.subheader(f"Prediction Result ({model_name})")
    if prediction == 1:
        st.error(f"The {model_name} model predicts a **HIGH** likelihood of lung cancer (Probability: {prediction_proba[1]:.2%}).")
        st.markdown("**Recommendation**: Please consult a healthcare professional immediately for further evaluation.")
    else:
        st.success(f"The {model_name} model predicts a **LOW** likelihood of lung cancer (Probability: {prediction_proba[1]:.2%}).")
        st.markdown("**Note**: Regular check-ups are recommended to maintain health.")

# Information about lung cancer
st.subheader("About Lung Cancer")
st.markdown("""
Lung cancer is a type of cancer that begins in the lungs, often linked to smoking or exposure to certain toxins.
### Risk Factors
- **Smoking**: The leading cause of lung cancer.
- **Secondhand Smoke**: Exposure to others' smoke increases risk.
- **Radon Gas**: A naturally occurring gas that can cause lung cancer.
- **Family History**: Genetic predisposition may increase risk.

### Symptoms
- Persistent cough
- Chest pain
- Shortness of breath
- Unexplained weight loss

### Prevention
- Quit smoking and avoid secondhand smoke.
- Test homes for radon.
- Maintain a healthy lifestyle with regular exercise and a balanced diet.

For more information, consult trusted medical resources or a healthcare provider.
""")

# Footer
st.markdown("---")
st.markdown("Developed using Streamlit and Machine Learning. Data inputs are used solely for prediction and not stored.")