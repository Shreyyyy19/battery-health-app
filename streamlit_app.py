import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# App config
st.set_page_config(page_title="📊 Battery Health Checker", layout="wide")
st.title("🔋 EV Battery Health Predictor")

# Load model and scalers
model = tf.keras.models.load_model("my_model.h5", compile=False)
y_min = float(np.load("y_scaler_min.npy"))
y_scale = float(np.load("y_scaler_scale.npy"))

st.markdown("Upload a **CSV file** with battery cycle data to predict health status based on Mean Voltage.")

uploaded_file = st.file_uploader("Upload Battery Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    required_cols = [
        "ambient_temperature", "mean_current", "mean_temperature",
        "max_voltage", "min_voltage", "max_current", "min_current",
        "mean_current_charge", "mean_voltage_charge"
    ]
    if not all(col in df.columns for col in required_cols):
        st.error("❌ CSV file is missing one or more required columns.")
        st.stop()

    # Format input
    X = df[required_cols].astype(np.float32)
    scaled_preds = model.predict(X).flatten()
    predicted_voltage = scaled_preds * y_scale + y_min

    df["Predicted_Mean_Voltage"] = predicted_voltage

    def classify_health(v):
        if v >= 3.6:
            return "Good"
        elif v >= 3.4:
            return "Warning"
        else:
            return "Needs Replacement"

    df["Battery_Health"] = df["Predicted_Mean_Voltage"].apply(classify_health)

    st.subheader("🔍 Prediction Results")
    st.dataframe(df)

    st.download_button("📥 Download Results CSV", df.to_csv(index=False), file_name="battery_predictions.csv")

    st.subheader("📈 Battery Health Overview")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=df.index, y="Predicted_Mean_Voltage", hue="Battery_Health", data=df, ax=ax, palette="Set2")
    ax.axhline(3.6, ls='--', c='green', label='Good Threshold')
    ax.axhline(3.4, ls='--', c='orange', label='Warning Threshold')
    ax.set_ylabel("Predicted Mean Voltage (V)")
    ax.set_xlabel("Battery Cycle Index")
    ax.set_title("Battery Health Prediction")
    st.pyplot(fig)
