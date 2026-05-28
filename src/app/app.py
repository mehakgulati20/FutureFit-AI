import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import time
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="FutureFit AI",
    page_icon="🩺",
    layout="wide"
)

from recommendations.diabetes import get_diabetes_report
from recommendations.cardio import cardio_recommendation
from recommendations.bp import bp_recommendation
from recommendations.cholesterol import cholesterol_recommendation

BASE_DIR = Path(__file__).resolve().parents[2]
IMAGE_PATH = BASE_DIR / "run.jpeg"

HF_REPO_ID = "mehakgulati20/FutureFit-AI-models"


@st.cache_resource
def load_model(filename):
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename
    )
    return joblib.load(model_path)


st.markdown("""
<style>
@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(25px);}
    to {opacity: 1; transform: translateY(0);}
}

.stApp {
    background: linear-gradient(135deg, #f3f4ff 0%, #eefcf8 50%, #fff7ed 100%);
}

.hero {
    padding: 50px;
    border-radius: 30px;
    background: linear-gradient(135deg, #2563eb, #14b8a6);
    color: white;
    animation: fadeInUp 0.8s ease;
    box-shadow: 0 20px 50px rgba(0,0,0,0.15);
}

.hero h1 {
    font-size: 58px;
    font-weight: 800;
}

.hero p {
    font-size: 20px;
    line-height: 1.7;
}

.badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 30px;
    background: rgba(255,255,255,0.2);
    font-weight: 700;
    margin-bottom: 18px;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 12px;
    color: #1e293b;
}

.metric-box {
    background: white;
    padding: 22px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    animation: fadeInUp 0.7s ease;
}

.metric-box h3 {
    color: #475569;
    font-size: 18px;
}

.metric-box p {
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
}

.result-card {
    background: white;
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    margin-bottom: 24px;
    animation: fadeInUp 0.8s ease;
}

.tip-box {
    background: #f8fafc;
    padding: 14px;
    border-radius: 16px;
    margin-bottom: 10px;
    border-left: 5px solid #14b8a6;
}

.stButton > button {
    width: 100%;
    border-radius: 18px;
    padding: 14px;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(135deg, #2563eb, #14b8a6);
    color: white;
    border: none;
}

img {
    border-radius: 24px;
}
</style>
""", unsafe_allow_html=True)


def set_bmi_category(input_data, bmi_value):
    if bmi_value < 18.5:
        col = "BMI_Category_Underweight"
    elif bmi_value < 25:
        col = "BMI_Category_Normal"
    elif bmi_value < 30:
        col = "BMI_Category_Overweight"
    else:
        col = "BMI_Category_Obese"

    if col in input_data.columns:
        input_data[col] = 1

    return input_data


def set_one_hot_column(input_data, column_name):
    if column_name in input_data.columns:
        input_data[column_name] = 1

    return input_data


left, right = st.columns([1.6, 1])

with left:
    st.markdown("""
    <div class="hero">
        <div class="badge">AI Powered Health Prediction</div>
        <h1>FutureFit AI</h1>
        <p>
        Predict diabetes, cholesterol, blood pressure, and cardiovascular risks
        using machine learning and personalized health analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    if IMAGE_PATH.exists():
        st.image(str(IMAGE_PATH), use_container_width=True)


with st.form("health_form"):
    st.markdown('<div class="section-title">🧍 Basic Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    with col3:
        smoking_history = st.selectbox(
            "Smoking History",
            ["Never", "Former", "Current", "Not Current", "Ever", "Unknown"]
        )

    st.markdown('<div class="section-title">📏 Body Measurements</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        feet = st.number_input("Height (ft)", min_value=0, max_value=8, value=5)

    with col2:
        inches = st.number_input("Height (in)", min_value=0, max_value=11, value=5)

    with col3:
        weight = st.number_input("Weight (lbs)", min_value=1, max_value=1000, value=120)

    st.markdown('<div class="section-title">🩸 Health Indicators</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        hbA1c = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=5.5)

    with col2:
        glucose = st.number_input("Glucose", min_value=50, max_value=300, value=120)

    with col3:
        ap_hi = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)

    with col4:
        ap_lo = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)

    st.markdown('<div class="section-title">🌱 Lifestyle Habits</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        smoking = st.radio("Do you smoke?", ["Yes", "No"], horizontal=True)

    with col2:
        alcohol = st.radio("Alcohol Consumption", ["Yes", "No"], horizontal=True)

    with col3:
        exercise = st.radio("Do you exercise?", ["Yes", "No"], horizontal=True)

    submit = st.form_submit_button("✨ Predict My Health Risks")


if submit:
    smoking_binary = 1 if smoking == "Yes" else 0
    alcohol_binary = 1 if alcohol == "Yes" else 0
    exercise_binary = 1 if exercise == "Yes" else 0

    height_inches = (feet * 12) + inches

    if height_inches == 0:
        st.error("Height cannot be zero.")
        st.stop()

    bmi = (weight * 703) / (height_inches ** 2)
    lifestyle_score = exercise_binary - alcohol_binary - smoking_binary

    with st.spinner("Analyzing your health profile..."):
        time.sleep(1.5)

    diabetes_model = load_model("diabetes_random_forest_model.pkl")
    bp_model = load_model("bp_model.pkl")
    cholesterol_model = load_model("cholesterol_model.pkl")
    cardio_model = load_model("cardio_model.pkl")

    diabetes_input = pd.DataFrame(columns=diabetes_model.feature_names_in_)
    diabetes_input.loc[0] = 0

    diabetes_input["age"] = age
    diabetes_input["bmi"] = bmi
    diabetes_input["HbA1c_level"] = hbA1c
    diabetes_input["blood_glucose_level"] = glucose

    diabetes_input = set_bmi_category(diabetes_input, bmi)
    diabetes_input = set_one_hot_column(diabetes_input, f"gender_{gender}")
    diabetes_input = set_one_hot_column(diabetes_input, f"smoking_history_{smoking_history}")

    diabetes_prediction = diabetes_model.predict(diabetes_input)[0]
    diabetes_output = "High" if diabetes_prediction == 1 else "Low"

    bp_input = pd.DataFrame([{
        "age_years": age,
        "BMI": bmi,
        "smoking": smoking_binary,
        "alcohol": alcohol_binary,
        "exercise": exercise_binary,
        "lifestyle_score": lifestyle_score,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo
    }])

    bp_prediction = bp_model.predict(bp_input)[0]

    if bp_prediction == 1:
        bp_output = "Normal"
    elif bp_prediction == 2:
        bp_output = "Elevated"
    elif bp_prediction == 3:
        bp_output = "Hypertension Stage 1"
    else:
        bp_output = "Hypertension Stage 2"

    cholesterol_input = pd.DataFrame([{
        "age_years": age,
        "BMI": bmi,
        "smoking": smoking_binary,
        "alcohol": alcohol_binary,
        "exercise": exercise_binary,
        "lifestyle_score": lifestyle_score
    }])

    cholesterol_prediction = cholesterol_model.predict(cholesterol_input)[0]

    if cholesterol_prediction == 1:
        cholesterol_output = "Normal"
    elif cholesterol_prediction == 2:
        cholesterol_output = "Above Normal"
    else:
        cholesterol_output = "Well Above Normal"

    cardio_input = pd.DataFrame([{
        "age_years": age,
        "BMI": bmi,
        "smoking": smoking_binary,
        "alcohol": alcohol_binary,
        "exercise": exercise_binary,
        "lifestyle_score": lifestyle_score
    }])

    cardio_prediction = cardio_model.predict(cardio_input)[0]
    cardio_output = "High" if cardio_prediction == 1 else "Low"

    st.markdown("## 📊 Health Report")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f'<div class="metric-box"><h3>Diabetes</h3><p>{diabetes_output}</p></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="metric-box"><h3>Blood Pressure</h3><p>{bp_output}</p></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="metric-box"><h3>Cholesterol</h3><p>{cholesterol_output}</p></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="metric-box"><h3>Cardio</h3><p>{cardio_output}</p></div>',
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            f'<div class="metric-box"><h3>BMI</h3><p>{bmi:.1f}</p></div>',
            unsafe_allow_html=True
        )

    cardio_recs = cardio_recommendation(cardio_prediction)
    cholesterol_recs = cholesterol_recommendation(cholesterol_prediction)
    bp_recs = bp_recommendation(bp_prediction - 1)

    diabetes_recs = get_diabetes_report(
        name="User",
        glucose=glucose,
        hba1c=hbA1c,
        bmi=bmi,
        age=age,
        hypertension=(bp_output != "Normal"),
        smoking=(smoking == "Yes")
    )

    reports = [
        ("🩸 Diabetes", diabetes_recs["label"], diabetes_recs["summary"], diabetes_recs["tips"]),
        ("❤️ Cardiovascular", cardio_recs["risk_level"], cardio_recs["summary"], cardio_recs["tips"]),
        ("🥗 Cholesterol", cholesterol_recs["risk_level"], cholesterol_recs["summary"], cholesterol_recs["tips"]),
        ("💓 Blood Pressure", bp_recs["risk_level"], bp_recs["summary"], bp_recs["tips"])
    ]

    for title, level, summary, tips in reports:
        st.markdown(
            f"""
            <div class="result-card">
                <h2>{title}</h2>
                <h3>{level}</h3>
                <p>{summary}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        for tip in tips:
            st.markdown(
                f"""
                <div class="tip-box">
                    • {tip}
                </div>
                """,
                unsafe_allow_html=True
            )
