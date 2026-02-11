import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stroke Prediction App", layout="centered")
st.title("🧠 Stroke Risk Predictor")
st.markdown("This app predicts the *likelihood of stroke* based on your health information.")

# Load model and encoders
try:
    model = joblib.load("stroke_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model or encoders: {e}")
    st.stop()

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.slider("Avg Glucose Level", 50.0, 300.0, 100.0)

    with col2:
        age = st.slider("Age", 1, 100, 30)
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

    # Encode categorical features
    try:
        for col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
    except Exception as e:
        st.error(f"❌ Error during label encoding: {e}")
        st.stop()

    # Predict
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # Medically informed risk interpretation thresholds
    if probability < 0.10:
        risk_level = "Low Risk ✅"
        advice = "Keep a healthy lifestyle and monitor your health regularly."
    elif probability < 0.25:
        risk_level = "Moderate Risk ⚠️"
        advice = "Consider lifestyle changes and consult your healthcare provider."
    else:
        risk_level = "High Risk 🚨"
        advice = "Please seek urgent medical evaluation and intervention."

    # Display results
    st.markdown("### 🧾 Prediction Result")
    st.write(f"🔍 Stroke Risk Level: *{risk_level}*")
    st.write(f"Probability of stroke: *{probability * 100:.2f}%*")
    st.info(advice)

    # Health overview bar chart
    st.markdown("### 📊 Health Overview")
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(input_df.columns, input_df.iloc[0], color='orange')
    plt.xticks(rotation=45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    st.pyplot(fig)

st.markdown("---")
st.markdown("Developed by **Andleeb Razzaq** — Stroke Prediction Project")
