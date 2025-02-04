import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor

# Load the trained model
with open("..\\models\\poisson_reg_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Diabetes Complications Prediction App 🏥")

st.write("Enter patient details to predict the expected number of complications.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=45)
gender = st.selectbox("Gender", ["Female", "Male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
total_comorbidities = st.number_input("Total Comorbidities", min_value=0, max_value=10, value=1)

# City selection (One-Hot Encoding, "Riyadh" is baseline)
city = st.selectbox("City", ['Riyadh', 'Other', 'Jeddah', 'Alkhobar', 'Makkah', 'Madina'])

# One-hot encoding for city
city_mapping = {'Other': 0, 'Jeddah': 0, 'Alkhobar': 0, 'Makkah': 0, 'Madina': 0}  # Default (Riyadh)
if city != "Riyadh":
    city_mapping[city] = 1  # Mark selected city as 1

# Gender encoding (Female is baseline)
gender_encoded = 1 if gender == "Male" else 0  # Male = 1, Female = 0 (baseline)

# Prepare data for prediction (Ensuring order matches model training)
input_data = np.array([[ 
    city_mapping['Other'], 
    city_mapping['Jeddah'], 
    city_mapping['Alkhobar'], 
    city_mapping['Makkah'], 
    city_mapping['Madina'], 
    age, 
    gender_encoded, 
    bmi, 
    total_comorbidities
]])

# Make prediction
if st.button("Predict"):
    st.text(input_data)
    prediction = model.predict(input_data)
    st.text(prediction)
    st.success(f"Predicted Number of Complications: {prediction[0]:.2f}")

