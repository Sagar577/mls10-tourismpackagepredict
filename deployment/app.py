import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# 1. Configuration - Matching your train.py setup
REPO_ID = "SagarAtHf/wellness-tourism-model-hub"
FILENAME = "productionmodel.joblib"

@st.cache_resource # This ensures the model only downloads once, not on every click
def load_model():
    try:
        # Pulling the model from the Model Hub
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model from Hub: {e}")
        return None

# Load the model
model = load_model()

# 2. UI Header
st.title("🌴 Wellness Tourism Package Predictor")
st.markdown("Enter customer details below to predict the likelihood of a package purchase.")

# 3. User Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        duration = st.number_input("Duration of Pitch", value=15)
        
    with col2:
        marital_status = st.selectbox("Marital Status", ["Married", "Unmarried", "Divorced"])
        designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
        product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
        monthly_income = st.number_input("Monthly Income", value=20000)
        passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    # Additional features required by the model (using averages/defaults)
    submit = st.form_submit_button("Predict Probability")

# 4. Prediction Logic
if submit and model:
    # Prepare input dataframe with exact column names from training
    input_data = pd.DataFrame({
        'Age': [age],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [2], # Defaulting common values
        'NumberOfFollowups': [3],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [3],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [1],
        'Passport': [passport],
        'PitchSatisfactionScore': [3],
        'OwnCar': [1],
        'NumberOfChildrenVisiting': [0],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income],
        'TypeofContact': ["Self Enquiry"]
    })

    # Get probability from the model
    # Note: We use the threshold 0.45 you defined in your local tests
    prob = model.predict_proba(input_data)[0][1]
    prediction = 1 if prob >= 0.45 else 0

    # 5. Display Results
    st.divider()
    if prediction == 1:
        st.success(f"🎯 High Potential Customer! (Probability: {prob:.2%})")
        st.balloons()
    else:
        st.warning(f"⏳ Low Likelihood of purchase. (Probability: {prob:.2%})")
