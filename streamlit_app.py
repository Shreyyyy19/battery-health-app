import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Battery Health Analyzer", layout="wide")
st.title("🔋 EV Battery Health Analyzer")

st.markdown("""
Upload your `.csv` file containing predicted battery health data.
The file should include at least the following columns:

- `battery_id`
- `cycle`
- `Predicted_Mean_Voltage`

The app will:
- Clean the data
- Plot voltage vs cycle per battery
- Classify health as Good, Warning, or Needs Replacement
""")

# --- Upload CSV ---
uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Data Preprocessing ---
    df = df.dropna()
    if 'Predicted_Mean_Voltage' not in df.columns:
        st.error("Missing column: 'Predicted_Mean_Voltage'")
        st.stop()
        
    df["Predicted_Mean_Voltage"] = df["Predicted_Mean_Voltage"].abs()  # fix negatives

    # --- Health Classification ---
    def classify(voltage):
        if voltage > 3.6:
            return "Good"
        elif voltage > 3.4:
            return "Warning"
        else:
            return "Needs Replacement"

    df["Health_Status"] = df["Predicted_Mean_Voltage"].apply(classify)

    # --- Summary Table with Colors ---
    st.subheader("🔍 Battery Health Table")
    status_colors = {
        "Good": "lightgreen",
        "Warning": "khaki",
        "Needs Replacement": "salmon"
    }

    styled_df = df.style.applymap(
        lambda val: f"background-color: {status_colors.get(val, '')}",
        subset=["Health_Status"]
    ).format({"Predicted_Mean_Voltage": "{:.3f}"})

    st.dataframe(styled_df, use_container_width=True)

    # --- Plot per Battery ---
    st.subheader("📈 Voltage Trends by Battery")

    unique_batteries = df["battery_id"].unique()
    for battery in unique_batteries:
        sub_df = df[df["battery_id"] == battery]
        plt.figure(figsize=(10, 4))
        sns.lineplot(x="cycle", y="Predicted_Mean_Voltage", data=sub_df, marker="o")
        plt.axhline(3.6, color="green", linestyle="--", label="Good Threshold (3.6V)")
        plt.axhline(3.4, color="orange", linestyle="--", label="Warning Threshold (3.4V)")
        plt.title(f"Battery ID: {battery}")
        plt.xlabel("Cycle")
        plt.ylabel("Predicted Mean Voltage (V)")
        plt.legend()
        st.pyplot(plt)
        plt.close()
