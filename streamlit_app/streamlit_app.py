import streamlit as st
import pickle
import math

# App 1 - Diabetes Complications Prediction
def diabetes_complications_prediction():
    # Load the trained model
    with open("poisson_reg_model.pkl", "rb") as f:
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

# App 2 - Charlson Comorbidity Index (CCI) Calculator
def charlson_comorbidity_index():
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

# Streamlit App - Main App with Tabs
def main():
    st.title("Health Data Applications")
    
    # Create tabs
    app_mode = st.radio("Select an app", ("Diabetes Complications Prediction", "Charlson Comorbidity Index (CCI) Calculator"))
    
    if app_mode == "Diabetes Complications Prediction":
        diabetes_complications_prediction()
    elif app_mode == "Charlson Comorbidity Index (CCI) Calculator":
        charlson_comorbidity_index()

if __name__ == "__main__":
    main()
