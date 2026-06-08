import streamlit as st
import joblib
import os
import pandas as pd

st.set_page_config(
    page_title="HealthRisk AI Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 HealthRisk AI Dashboard")

# Load models
try:
    heart_model = joblib.load("models/heart_model.pkl")
    insurance_model = joblib.load("models/insurance_model.pkl")
    hospital_model = joblib.load("models/hospital_model.pkl")
    pharma_model = joblib.load("models/pharma_model.pkl")

    st.success("✅ All Models Loaded Successfully")

except Exception as e:
    st.error(f"Error Loading Models: {e}")

# Sidebar
option = st.sidebar.selectbox(
    "Choose Prediction Module",
    [
        "Home",
        "Heart Disease Prediction",
        "Insurance Risk Prediction",
        "Hospital Risk Prediction",
        "Pharma Recommendation",
        "HealthRisk Lab Simulator"
    ]
)
# Home Page
if option == "Home":

    st.header("Welcome to HealthRisk AI")

    st.write("""
    This dashboard contains:

    • Heart Disease Prediction

    • Insurance Risk Prediction

    • Hospital Risk Prediction

    • Pharma Recommendation System
    """)

# Heart Page
elif option == "Heart Disease Prediction":

    st.header("❤️ Heart Disease Prediction")

    Age = st.number_input("Age", 18, 100, 40)

    Sex = st.selectbox("Sex", ["M", "F"])

    ChestPainType = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    RestingBP = st.number_input("Resting BP", 80, 250, 120)

    Cholesterol = st.number_input(
        "Cholesterol",
        0,
        700,
        200
    )

    FastingBS = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1]
    )

    RestingECG = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    MaxHR = st.number_input(
        "Max Heart Rate",
        60,
        250,
        150
    )

    ExerciseAngina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"]
    )

    Oldpeak = st.number_input(
        "Oldpeak",
        0.0,
        10.0,
        1.0
    )

    ST_Slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    if st.button("Predict Heart Disease"):

        sex_map = {"F": 0, "M": 1}
        cp_map = {
            "ASY": 0,
            "ATA": 1,
            "NAP": 2,
            "TA": 3
        }
        ecg_map = {
            "LVH": 0,
            "Normal": 1,
            "ST": 2
        }
        angina_map = {
            "N": 0,
            "Y": 1
        }
        slope_map = {
            "Down": 0,
            "Flat": 1,
            "Up": 2
        }

        prediction = heart_model.predict([[
            Age,
            sex_map[Sex],
            cp_map[ChestPainType],
            RestingBP,
            Cholesterol,
            FastingBS,
            ecg_map[RestingECG],
            MaxHR,
            angina_map[ExerciseAngina],
            Oldpeak,
            slope_map[ST_Slope]
        ]])

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")
# Insurance Page
elif option == "Insurance Risk Prediction":

    st.header("🛡 Insurance Risk Prediction")

    age = st.number_input("Age", 18, 100, 30)

    sex = st.selectbox(
        "Sex",
        ["female", "male"]
    )

    bmi = st.number_input(
        "BMI",
        10.0,
        60.0,
        25.0
    )

    children = st.number_input(
        "Children",
        0,
        10,
        0
    )

    smoker = st.selectbox(
        "Smoker",
        ["no", "yes"]
    )

    region = st.selectbox(
        "Region",
        [
            "southwest",
            "southeast",
            "northwest",
            "northeast"
        ]
    )

    if st.button("Predict Insurance Risk"):

        sex_map = {
            "female": 0,
            "male": 1
        }

        smoker_map = {
            "no": 0,
            "yes": 1
        }

        region_map = {
            "northeast": 0,
            "northwest": 1,
            "southeast": 2,
            "southwest": 3
        }

        prediction = insurance_model.predict([[
            age,
            sex_map[sex],
            bmi,
            children,
            smoker_map[smoker],
            region_map[region]
        ]])

        st.success(
            f"Predicted Insurance Risk: {prediction[0]}"
        )
# Hospital Page
elif option == "Hospital Risk Prediction":

    st.header("🏥 Hospital Risk Prediction")

    revenue = st.number_input(
        "Revenue",
        min_value=0.0,
        value=1000000.0
    )

    debt_ratio = st.number_input(
        "Debt Ratio",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

    readmission_rate = st.number_input(
        "Readmission Rate",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )

    patient_satisfaction = st.number_input(
        "Patient Satisfaction",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    if st.button("Predict Hospital Risk"):

        prediction = hospital_model.predict([[
            revenue,
            debt_ratio,
            readmission_rate,
            patient_satisfaction
        ]])

        st.success(
            f"Predicted Hospital Risk: {prediction[0]}"
        )
# Pharma Page
elif option == "Pharma Recommendation":

    st.header("💊 Pharma Recommendation")

    rd_spend = st.number_input(
        "R&D Spend",
        min_value=0.0,
        value=500000.0
    )

    revenue_growth = st.number_input(
        "Revenue Growth (%)",
        min_value=-100.0,
        max_value=500.0,
        value=10.0
    )

    trial_success = st.number_input(
        "Trial Success Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    if st.button("Get Recommendation"):

        prediction = pharma_model.predict([[
            rd_spend,
            revenue_growth,
            trial_success
        ]])

        st.success(
            f"Recommendation: {prediction[0]}"
        )
elif option == "HealthRisk Lab Simulator":

    st.header("🧪 HealthRisk Lab Simulator")
    st.write("Change inputs and run all 4 AI models together.")

    st.subheader("Patient Profile")

    age = st.number_input("Age", 18, 100, 40)
    sex = st.selectbox("Gender", ["M", "F"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    cholesterol = st.number_input("Cholesterol", 0, 700, 200)
    resting_bp = st.number_input("Resting BP", 80, 250, 120)

    st.subheader("Hospital Metrics")

    revenue = st.number_input(
        "Hospital Revenue",
        min_value=0.0,
        value=1000000.0
    )

    debt_ratio = st.number_input(
        "Debt Ratio",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

    readmission_rate = st.number_input(
        "Readmission Rate",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )

    patient_satisfaction = st.number_input(
        "Patient Satisfaction",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    st.subheader("Pharma Metrics")

    rd_spend = st.number_input(
        "R&D Spend",
        min_value=0.0,
        value=500000.0
    )

    revenue_growth = st.number_input(
        "Revenue Growth (%)",
        min_value=-100.0,
        max_value=500.0,
        value=10.0
    )

    trial_success = st.number_input(
        "Trial Success Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    if st.button("🚀 Run Full Simulation"):

        sex_map = {"F": 0, "M": 1}

        heart_prediction = heart_model.predict([[
            age,
            sex_map[sex],
            1,
            resting_bp,
            cholesterol,
            0,
            1,
            150,
            0,
            1.0,
            2
        ]])

        insurance_prediction = insurance_model.predict([[
            age,
            sex_map[sex],
            bmi,
            0,
            1 if smoker == "yes" else 0,
            0
        ]])

        hospital_prediction = hospital_model.predict([[
            revenue,
            debt_ratio,
            readmission_rate,
            patient_satisfaction
        ]])

        pharma_prediction = pharma_model.predict([[
            rd_spend,
            revenue_growth,
            trial_success
        ]])

        st.subheader("📊 Simulation Results")

        if heart_prediction[0] == 1:
            st.error("❤️ Heart Risk: HIGH")
        else:
            st.success("❤️ Heart Risk: LOW")

        st.info(f"🛡 Insurance Risk: {insurance_prediction[0]}")
        st.info(f"🏥 Hospital Risk: {hospital_prediction[0]}")
        st.info(f"💊 Pharma Recommendation: {pharma_prediction[0]}")

        risk_score = 0

        if heart_prediction[0] == 1:
            risk_score += 25

        if smoker == "yes":
            risk_score += 25

        if bmi > 30:
            risk_score += 25

        if cholesterol > 240:
            risk_score += 25
        st.subheader("📈 Overall Health Risk Score")
        st.progress(risk_score / 100)
        st.metric("Risk Score", f"{risk_score}%")
        if risk_score >= 75:
            st.error("🔥 HIGH RISK ALERT")
        elif risk_score >= 50:
            st.warning("⚠️ MODERATE RISK ALERT")
        else:
            st.success("✅ LOW RISK")
        chart_data = pd.DataFrame({
            "Metric": ["Heart", "Insurance", "Hospital", "Pharma"],
            "Value": [100 if heart_prediction[0] == 1 else 25, 50, 50, 50]
        })

        st.subheader("📊 Risk Comparison Graph")
        st.bar_chart(chart_data.set_index("Metric"))

        st.subheader("📋 Executive Summary")

        report = f"""
HEALTHRISK AI REPORT

Heart Risk:
{'High' if heart_prediction[0] == 1 else 'Low'}

Insurance Risk:
{insurance_prediction[0]}

Hospital Risk:
{hospital_prediction[0]}

Pharma Recommendation:
{pharma_prediction[0]}

Overall Risk Score:
{risk_score}%
"""

        st.download_button(
            label="💾 Download Report",
            data=report,
            file_name="healthrisk_report.txt",
            mime="text/plain"
        )

        st.write(f"""
        - Heart Risk: {'High' if heart_prediction[0] == 1 else 'Low'}
        - Insurance Risk: {insurance_prediction[0]}
        - Hospital Risk: {hospital_prediction[0]}
        - Pharma Recommendation: {pharma_prediction[0]}
        - Overall Risk Score: {risk_score}%
        """)