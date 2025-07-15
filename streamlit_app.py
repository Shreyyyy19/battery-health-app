import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load model and scalers
model = tf.keras.models.load_model("my_model.h5", compile=False)
y_min = np.load("y_scaler_min.npy")
y_scale = np.load("y_scaler_scale.npy")

st.set_page_config(page_title="EV Battery Health Predictor", layout="centered")
st.title("🔋 EV Battery Health Predictor")
st.markdown("Use this tool to **upload EV battery data**, run predictions, and visualize battery health over time.")

uploaded_file = st.file_uploader("Upload Battery Cycle Data (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_features = [
        "ambient_temperature", "mean_current", "mean_temperature",
        "max_voltage", "min_voltage", "max_current", "min_current",
        "mean_current_charge", "mean_voltage_charge"
    ]

    if not all(col in df.columns for col in required_features):
        st.error(f"CSV must contain columns: {', '.join(required_features)}")
    else:
        X = df[required_features]
        predictions_scaled = model.predict(X)
        predictions = predictions_scaled * y_scale + y_min
        predictions = predictions.flatten()

        df['Predicted_Mean_Voltage'] = predictions
        df['Health_Status'] = pd.cut(
            df['Predicted_Mean_Voltage'],
            bins=[0, 3.4, 3.6, 5],
            labels=['Needs Replacement', 'Warning', 'Good']
        )

        st.success("✅ Prediction complete!")

        st.dataframe(df[['Predicted_Mean_Voltage', 'Health_Status']])

        st.download_button("📥 Download Results as CSV", df.to_csv(index=False), file_name="battery_health_predictions.csv")

        import matplotlib.pyplot as plt
        import seaborn as sns

        df['Cycle'] = range(1, len(df)+1)
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x='Cycle', y='Predicted_Mean_Voltage', label='Mean Voltage', linewidth=2)
        plt.axhline(3.6, color='orange', linestyle='--', label='Warning Threshold (3.6V)')
        plt.axhline(3.4, color='red', linestyle='--', label='Replacement Threshold (3.4V)')
        plt.xlabel("Cycle")
        plt.ylabel("Predicted Mean Voltage (V)")
        plt.title("Predicted Mean Voltage Over Battery Cycles")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

else:
    st.info("⬆️ Upload a CSV file to get started. You can export sample cycle data from NASA or use your own.")
