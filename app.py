import streamlit as st
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Multimodal Mental Health Predictor", page_icon="🧠")

# Class label mapping from training
CLASS_NAMES = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality Disorder",
    5: "Stress",
    6: "Suicidal"
}

# Severity weights for smarter fusion
SEVERITY_WEIGHT = {
    "Normal": 0.0,
    "Stress": 0.2,
    "Anxiety": 0.4,
    "Depression": 0.6,
    "Bipolar": 0.6,
    "Personality Disorder": 0.7,
    "Suicidal": 1.0
}

# ---------------- LOAD MODELS (cached) ----------------
@st.cache_resource
def load_models():
    depression_model = joblib.load("tabular_models/depression_model.pkl")
    anxiety_model = joblib.load("tabular_models/anxiety_model.pkl")
    panic_model = joblib.load("tabular_models/panic_model.pkl")

    tokenizer = AutoTokenizer.from_pretrained("distilbert_model")
    bert_model = AutoModelForSequenceClassification.from_pretrained("distilbert_model")
    bert_model.eval()

    return depression_model, anxiety_model, panic_model, tokenizer, bert_model


depression_model, anxiety_model, panic_model, tokenizer, bert_model = load_models()

# ---------------- UI ----------------
st.title(" Multimodal Mental Health Risk Predictor")

st.markdown(
    """
This AI system combines **student survey data** and **chat message analysis**
to estimate **mental-health risk level**.

⚠️ *This tool is for educational screening only — not a medical diagnosis.*
"""
)

# ================= TABULAR INPUT =================
st.header("📋 Student Survey")

gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
age = st.slider("Age", 16, 35, 20)
course = st.selectbox("Course Code", list(range(0, 10)), help="Lower number = more difficult course, higher number = easier course")
year = st.selectbox("Year of Study", [1, 2, 3, 4])
cgpa = st.slider("CGPA", 0.0, 4.0, 3.0, step=0.01)
marital_status = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
treatment = st.selectbox("Seeking Treatment?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

st.subheader("Self-Reported Symptoms")
self_depression = st.selectbox("Do you feel depressed?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
self_anxiety = st.selectbox("Do you feel anxious?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
self_panic = st.selectbox("Do you experience panic attacks?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Feature DataFrames
dep_features = pd.DataFrame([[gender, age, course, year, cgpa, marital_status, self_anxiety, self_panic, treatment]],
                            columns=['gender', 'age', 'course', 'year', 'cgpa', 'marital_status', 'anxiety', 'panic_attack', 'treatment'])

anx_features = pd.DataFrame([[gender, age, course, year, cgpa, marital_status, self_depression, self_panic, treatment]],
                            columns=['gender', 'age', 'course', 'year', 'cgpa', 'marital_status', 'depression', 'panic_attack', 'treatment'])

pan_features = pd.DataFrame([[gender, age, course, year, cgpa, marital_status, self_depression, self_anxiety, treatment]],
                            columns=['gender', 'age', 'course', 'year', 'cgpa', 'marital_status', 'depression', 'anxiety', 'treatment'])

# Predictions
dep_prob = depression_model.predict_proba(dep_features)[0][1]
anx_prob = anxiety_model.predict_proba(anx_features)[0][1]
pan_prob = panic_model.predict_proba(pan_features)[0][1]

tabular_risk = 0.4 * dep_prob + 0.35 * anx_prob + 0.25 * pan_prob

st.info(f"**Tabular Risk Score:** {tabular_risk:.2f}")

# ================= TEXT INPUT =================
st.header("💬 Chat Message Analysis")

user_text = st.text_area("Enter student's message")

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    class_name = CLASS_NAMES[pred]

    return class_name, confidence, probs


if st.button("🔍 Analyze Mental Health"):

    if user_text.strip() == "":
        st.warning("Please enter a message.")
        st.stop()

    # -------- TEXT PREDICTION --------
    class_name, conf, probs = predict_text(user_text)

    # -------- SUICIDAL SAFETY KEYWORD CHECK --------
    suicidal_keywords = [
        "no reason to live",
        "want to die",
        "end my life",
        "better off without me",
        "can't go on",
        "kill myself",
        "suicide",
        "not worth living",
        "handle this pain",
        "die",
        "kill",
        "murder"
    ]

    text_lower = user_text.lower()
    suicidal_flag = any(keyword in text_lower for keyword in suicidal_keywords)

    if suicidal_flag:
        class_name = "Suicidal"

    st.subheader("Text Analysis Result")
    st.write(f"**Detected State:** {class_name}")
    st.write(f"**Model Confidence:** {conf:.2f}")

    # -------- SHOW CLASS PROBABILITY BAR CHART (skip for suicidal) --------
    if not suicidal_flag and class_name != "Suicidal":
        st.markdown("### 📊 Prediction Confidence Across Mental-Health Classes")

        prob_df = pd.DataFrame({
            "Mental State": list(CLASS_NAMES.values()),
            "Probability": probs
        })

        st.bar_chart(prob_df.set_index("Mental State"))

    # -------- SAFETY-AWARE FINAL SCORE --------
    if class_name == "Suicidal" or suicidal_flag:
        final_score = 1.0  # force HIGH risk
        st.error("🚨 High suicidal risk indicators detected. Immediate professional support is strongly recommended.")
    else:
        # severity-aware fusion (better than raw confidence)
        severity = SEVERITY_WEIGHT[class_name]
        final_score = 0.6 * tabular_risk + 0.4 * severity

    # -------- FINAL RESULT --------
    st.subheader(f"Final Mental-Health Risk Score: {final_score:.2f}")

    if final_score < 0.33:
        st.success("🟢 Low Risk")
    elif final_score < 0.66:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")

    # -------- RISK-BASED SUGGESTIONS --------
    st.markdown("### 🩺 Suggested Guidance")

    # Extra safety check for suicidal prediction
    if class_name == "Suicidal":
        st.error(
            """
🚨 **Possible Suicidal Risk Detected**

If you are in immediate danger or thinking about self-harm:

- Contact a **local emergency number** or **mental-health helpline** immediately.
- Reach out to someone you trust right now.
- Professional help can provide real support and protection.

Your safety matters.
"""
        )

    if final_score < 0.33:
        st.success(
            """
**Low Risk Detected**

- Your responses indicate generally healthy emotional well-being.
- Continue maintaining:
  - Regular sleep and physical activity
  - Social interaction with friends/family
  - Balanced academic routine
- Practice mindfulness or relaxation to stay mentally strong.
"""
        )

    elif final_score < 0.66:
        st.warning(
            """
**Moderate Risk Detected**

- Some signs of emotional stress or mental strain are present.
- Recommended actions:
  - Talk with a trusted friend, mentor, or family member.
  - Consider speaking with a **college counselor or psychologist**.
  - Reduce academic overload and maintain a daily routine.
  - Practice stress-management techniques (breathing, journaling, exercise).

Early support can significantly improve well-being.
"""
        )

    else:
        st.error(
            """
**High Mental-Health Risk Detected**

- Your responses suggest **significant emotional distress**.
- **Professional help is strongly recommended.**

Immediate steps:
- Contact a **mental-health professional, counselor, or doctor**.
- Reach out to a **trusted person** and avoid staying alone.
- If you feel unsafe, seek **urgent local medical support**.

You are not alone — support is available and recovery is possible.
"""
        )
