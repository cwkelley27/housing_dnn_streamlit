import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

st.set_page_config(page_title="Hamilton County Housing Value Predictor")

st.title("Hamilton County Housing Value Predictor")
st.caption("Educational use only. Predictions are approximate.")

# Load trained artifacts
model = tf.keras.models.load_model("artifacts/housing_model.h5", compile=False)
scaler = joblib.load("artifacts/scaler.pkl")
feature_names = joblib.load("artifacts/feature_names.pkl")

# User inputs
acres = st.number_input(
    "Land area (acres)",
    min_value=0.01,
    max_value=20.0,
    value=0.25,
    step=0.01
)

yearbuilt = st.number_input(
    "Year built",
    min_value=1900,
    max_value=2026,
    value=2000,
    step=1
)

sizearea = st.number_input(
    "Building area (sq ft)",
    min_value=300,
    max_value=10000,
    value=1800,
    step=50
)

if st.button("Predict"):
    # 1. Create a dictionary with ALL expected features set to 0.0
    input_data = {col: 0.0 for col in feature_names}

    # 2. Overwrite the specific numerical values the user provided
    # (Ensure these keys exactly match the numerical column names from your training data)
    if "CALC_ACRES" in input_data:
        input_data["CALC_ACRES"] = acres
    if "YEARBUILT" in input_data:
        input_data["YEARBUILT"] = yearbuilt
    if "SIZEAREA" in input_data:
        input_data["SIZEAREA"] = sizearea

    # 3. Convert to DataFrame using the EXACT column order the model expects
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # 4. Now the scaler sees the exact shape it expects!
    input_scaled = scaler.transform(input_df)

    # 5. Predict
    pred_value = model.predict(input_scaled, verbose=0)[0][0]

    st.success(f"Estimated appraised value: ${pred_value:,.0f}")

st.caption(
    "This app is for teaching and learning only. "
    "It is not suitable for real appraisal, lending, tax, or investment decisions."
)
