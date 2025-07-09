import streamlit as st
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("my_model.h5")

st.title("EV Battery Mean Voltage Predictor 🔋")

st.markdown("""
Use the sliders and inputs below to simulate one EV battery charge/discharge cycle.
The model will predict the **mean voltage** for that cycle based on selected parameters.
""")

ambient_temperature = st.number_input("Ambient Temperature (°C)", 20.0, 40.0, 25.0)
mean_current = st.number_input("Mean Current (A)", -5.0, 5.0, 0.5)
mean_temperature = st.number_input("Mean Cell Temperature (°C)", 20.0, 45.0, 30.0)
max_voltage = st.number_input("Max Voltage (V)", 3.5, 4.5, 4.2)
min_voltage = st.number_input("Min Voltage (V)", 2.5, 4.0, 3.3)
max_current = st.number_input("Max Current (A)", -5.0, 5.0, 1.5)
min_current = st.number_input("Min Current (A)", -5.0, 5.0, -3.0)
mean_current_charge = st.number_input("Mean Current Charge (A)", -5.0, 5.0, 0.8)
mean_voltage_charge = st.number_input("Mean Voltage Charge (V)", 3.0, 4.5, 4.1)

# form a single-row feature array
input_features = np.array([[ambient_temperature,
                            mean_current,
                            mean_temperature,
                            max_voltage,
                            min_voltage,
                            max_current,
                            min_current,
                            mean_current_charge,
                            mean_voltage_charge]], dtype=np.float32)

if st.button("Predict Mean Voltage"):
    prediction = model.predict(input_features)[0][0]
    st.success(f"🔌 Predicted Mean Voltage: **{prediction:.3f} V**")
