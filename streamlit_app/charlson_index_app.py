import streamlit as st
import math

# Charlson Comorbidity Weights
comorbidity_weights = {
    "Myocardial Infarction": 1,
    "Congestive Heart Failure": 1,
    "Peripheral Vascular Disease": 1,
    "Cerebrovascular Disease": 1,
    "Dementia": 1,
    "Chronic Pulmonary Disease": 1,
    "Rheumatologic Disease": 1,
    "Peptic Ulcer Disease": 1,
    "Mild Liver Disease": 1,
    "Diabetes (without complications)": 1,
    "Diabetes (End-organ damage)": 2,
    "Hemiplegia or Paraplegia": 2,
    "Moderate or Severe Renal Disease": 2,
    "Any Tumor (excluding skin cancers)": 2,
    "Metastatic Solid Tumor": 6,
    "Leukemia": 2,
    "Lymphoma": 2,
    "Moderate or Severe Liver Disease": 3,
    "AIDS/HIV": 6,
}

def calculate_cci(age, selected_conditions):
    # Age adjustment
    if 50 <= age < 60:
        age_score = 1
    elif 60 <= age < 70:
        age_score = 2
    elif 70 <= age < 80:
        age_score = 3
    elif age >= 80:
        age_score = 4
    else:
        age_score = 0
    
    # Calculate total score
    cci_score = sum(comorbidity_weights.get(cond, 0) for cond in selected_conditions) + age_score
    return cci_score

def calculate_10yr_survival(cci_score):
    return round(0.983 ** math.exp(cci_score * 0.9), 4)

# Streamlit UI
st.title("Charlson Comorbidity Index (CCI) Calculator")
st.write("Calculate the Charlson Comorbidity Index score based on patient conditions and age.")

# User Inputs
age = st.number_input("Enter Patient's Age", min_value=0, max_value=120, value=50)
conditions = list(comorbidity_weights.keys())
selected_conditions = st.multiselect("Select Comorbidities", options=conditions)

# Compute Score
if st.button("Calculate CCI Score"):
    cci_score = calculate_cci(age, selected_conditions)
    survival_rate = calculate_10yr_survival(cci_score)
    st.success(f"Charlson Comorbidity Index Score: {cci_score}")
    st.info(f"Estimated 10-year survival probability: {survival_rate * 100}%")
    
    # Interpretation
    if cci_score == 0:
        st.info("Low risk.")
    elif 1 <= cci_score <= 3:
        st.warning("Moderate risk.")
    else:
        st.error("High risk.")
