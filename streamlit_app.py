import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# App config
st.set_page_config(page_title="🔋 Battery Health Predictor", layout="wide")
st.title("🔋 EV Battery Health Predictor")

# Load model and scalers
model = tf.keras.models.load_model("my_model.h5", compile=False)
y_min = float(np.load("y_scaler_min.npy"))
y_scale = float(np.load("y_scaler_scale.npy"))

st.markdown("Upload a **CSV file** with battery cycle data. We'll predict mean voltage and classify battery health.")

uploaded_file = st.file_uploader("📂 Upload Battery Dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Required features
    required_cols = [
        "ambient_temperature", "mean_current", "mean_temperature",
        "max_voltage", "min_voltage", "max_current", "min_current",
        "mean_current_charge", "mean_voltage_charge"
    ]

    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Expected: {', '.join(required_cols)}")
        st.stop()

    # Drop rows with missing inputs
    clean_df = df.dropna(subset=required_cols)

    if clean_df.empty:
        st.warning("⚠️ All rows had missing input fields. Please check your file.")
        st.stop()

    # Make predictions
    X = clean_df[required_cols].astype(np.float32)
    scaled_preds = model.predict(X).flatten()
    predicted_voltage = scaled_preds * y_scale + y_min

    clean_df["Predicted_Mean_Voltage"] = predicted_voltage

    # Health classification
    def classify_health(v):
        if v >= 3.6:
            return "Good"
        elif v >= 3.4:
            return "Warning"
        else:
            return "Needs Replacement"

    clean_df["Battery_Health"] = clean_df["Predicted_Mean_Voltage"].apply(classify_health)

    # Color style for health column
    def color_health(val):
        if val == "Good":
            return "background-color: #d4edda; color: #155724"
        elif val == "Warning":
            return "background-color: #fff3cd; color: #856404"
        else:
            return "background-color: #f8d7da; color: #721c24"

    st.subheader("📊 Prediction Results")
    styled_table = clean_df.style.applymap(color_health, subset=["Battery_Health"])
    st.dataframe(styled_table)

    st.download_button("📥 Download CSV Results", clean_df.to_csv(index=False), file_name="battery_health_results.csv")

    # Visualization
    st.subheader("🔍 Visual Overview")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=clean_df.index, y="Predicted_Mean_Voltage", hue="Battery_Health", data=clean_df, dodge=False)

    if "battery_id" in clean_df.columns:
        ax.set_xticks(clean_df.index)
        ax.set_xticklabels(clean_df["battery_id"], rotation=90)

    ax.axhline(3.6, ls='--', c='green', label="Good Threshold")
    ax.axhline(3.4, ls='--', c='orange', label="Warning Threshold")
    ax.set_ylabel("Predicted Mean Voltage (V)")
    ax.set_title("Battery Health Classification by Cycle")
    ax.legend()
    st.pyplot(fig)
