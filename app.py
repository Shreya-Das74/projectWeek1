# app.py  (place this file in the same folder as:
# ckd_model.pkl, ckd_schema.pkl, X_train_background.pkl)

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------- Load saved artifacts --------------------
clf = joblib.load("ckd_model.pkl")
schema = joblib.load("ckd_schema.pkl")
X_bg_raw = joblib.load("X_train_background.pkl")

required_cols = schema["required_cols"]
defaults = schema["defaults"]

# -------------------- Page + styling --------------------
st.set_page_config(page_title="Chronic Kidney Risk Predictor", layout="wide")

st.markdown(
    """
    <style>
      .main { background-color: #0b0f17; }
      h1, h2, h3, label, p, div { color: #e9eef7; }
      .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        border: 0;
      }
      .card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 18px 8px 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
      }
      .ok {
        background: rgba(34,197,94,0.18);
        border: 1px solid rgba(34,197,94,0.35);
        padding: 16px;
        border-radius: 14px;
        font-size: 18px;
        font-weight: 700;
      }
      .warn {
        background: rgba(245,158,11,0.18);
        border: 1px solid rgba(245,158,11,0.35);
        padding: 16px;
        border-radius: 14px;
        font-size: 18px;
        font-weight: 700;
      }
      .bad {
        background: rgba(239,68,68,0.18);
        border: 1px solid rgba(239,68,68,0.35);
        padding: 16px;
        border-radius: 14px;
        font-size: 18px;
        font-weight: 700;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Chronic Kidney Risk Predictor")
st.caption("Enter patient details → get CKD risk (%) + AI explanation .")

# -------------------- Helpers --------------------
def get_default(col, fallback):
    try:
        return defaults[col]
    except KeyError:
        return fallback

def risk_label(risk_pct: float):
    if risk_pct < 33:
        return "Low Risk", "ok"
    if risk_pct < 66:
        return "Moderate Risk", "warn"
    return "High Risk", "bad"

# -------------------- Layout --------------------
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient Inputs")

    # NOTE: these must match your dataset column names
    age = st.number_input("Age", 1, 120, int(get_default("Age", 40)))
    systolic = st.number_input("SystolicBP", 50, 250, int(get_default("SystolicBP", 120)))
    creatinine = st.number_input("SerumCreatinine", 0.1, 20.0, float(get_default("SerumCreatinine", 1.2)))
    hemoglobin = st.number_input("HemoglobinLevels", 1.0, 20.0, float(get_default("HemoglobinLevels", 13.0)))

    if "Gender" in required_cols:
        gdef = str(get_default("Gender", "Male")).strip()
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if gdef.lower().startswith("m") else 1)
    else:
        gender = None

    predict_btn = st.button("Predict")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Output")

    if predict_btn:
        # 1) Build full row with ALL required cols (defaults + user overrides)
        row = {col: defaults[col] for col in required_cols}

        if "Age" in row: row["Age"] = int(age)
        if "SystolicBP" in row: row["SystolicBP"] = int(systolic)
        if "SerumCreatinine" in row: row["SerumCreatinine"] = float(creatinine)
        if "HemoglobinLevels" in row: row["HemoglobinLevels"] = float(hemoglobin)
        if gender is not None and "Gender" in row: row["Gender"] = str(gender).strip()

        input_data = pd.DataFrame([row], columns=required_cols)

        # 2) Risk prediction
        risk = float(clf.predict_proba(input_data)[0, 1] * 100.0)
        label, css = risk_label(risk)
        st.markdown(f'<div class="{css}">CKD Risk: {risk:.2f}%  •  {label}</div>', unsafe_allow_html=True)

        # 3) SHAP explanation (REAL background)
        preprocess = clf.named_steps["preprocess"]
        model = clf.named_steps["model"]
        feature_names = preprocess.get_feature_names_out()

        # sample background for speed
        bg = X_bg_raw.sample(n=min(300, len(X_bg_raw)), random_state=42)
        X_bg = preprocess.transform(bg)
        X_bg_df = pd.DataFrame(X_bg, columns=feature_names)

        X_row = preprocess.transform(input_data)
        X_row_df = pd.DataFrame(X_row, columns=feature_names)

        explainer = shap.TreeExplainer(model, data=X_bg_df)
        sv = explainer(X_row_df, check_additivity=False)

        st.subheader("AI Explanation (Why this risk?)")
        fig = plt.figure()
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Fill the inputs and click **Predict** to see the CKD risk and explanation.")

    st.markdown("</div>", unsafe_allow_html=True)