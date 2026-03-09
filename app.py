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
    input_df = pd.DataFrame({
        "CALC_ACRES": [acres],
        "YEARBUILT": [yearbuilt],
        "SIZEAREA": [sizearea]
    })

    input_scaled = scaler.transform(input_df)

    pred_value = model.predict(input_scaled, verbose=0)[0][0]

    st.success(f"Estimated appraised value: ${pred_value:,.0f}")

st.caption(
    "This app is for teaching and learning only. "
    "It is not suitable for real appraisal, lending, tax, or investment decisions."
)
