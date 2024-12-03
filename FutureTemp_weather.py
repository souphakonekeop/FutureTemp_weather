import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the trained model
model = load_model('lstm_weather_model.h5')

# Streamlit Configuration
st.set_page_config(page_title="Seattle Weather Predictor", page_icon="🌤️", layout="wide")

# Custom CSS for UI styling
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #2c3e50;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 8px 20px;
        font-size: 16px;
        margin: 10px 0px;
    }
    .stMetric {
        font-weight: bold;
    }
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.title("🌤️ FutureTemp Weather Predictor")
st.markdown("""
Welcome to the **FutureTemp**! This tool uses **LSTM Neural Networks** to forecast the temperature based on historical data. 
Enjoy a visually appealing and interactive experience. 🚀
""")

# Sidebar Section for Input
st.sidebar.header("🔧 Configure Inputs")
st.sidebar.markdown("Adjust the input parameters below:")
window_size = st.sidebar.slider("Number of Days for Prediction", min_value=5, max_value=20, value=10)
inputs = []
for i in range(window_size):
    inputs.append(st.sidebar.number_input(f"Day {i+1} Temperature (°C):", value=10.0))

# Prediction and Metrics Section
if st.sidebar.button("🌡️ Predict Temperature"):
    input_data = np.array(inputs).reshape(1, -1, 1)
    prediction = model.predict(input_data)[0][0]

    # Simulate actual temperature for metrics (replace with real data if available)
    actual_temp = [input_data[0, -1, 0] + np.random.uniform(-2, 2)]

    # Calculate Metrics
    mae = mean_absolute_error(actual_temp, [prediction])
    rmse = np.sqrt(mean_squared_error(actual_temp, [prediction]))
    accuracy = 100 - (abs(actual_temp[0] - prediction) / abs(actual_temp[0]) * 100)

    # Result Cards
    st.markdown("## 📊 Results")
    st.success(f"🌡️ **Predicted Temperature**: {prediction:.2f} °C")
    st.info(f"📏 **Simulated Actual Temperature**: {actual_temp[0]:.2f} °C")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} °C")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} °C")
    with col3:
        st.metric("Prediction Accuracy (%)", f"{accuracy:.2f} %")

    # Graph Section
    st.markdown("## 📈 Temperature Visualization")
    df_plot = pd.DataFrame({
        'Day': range(len(inputs) + 1),
        'Temperature': inputs + [actual_temp[0]],
        'Type': ['Input'] * len(inputs) + ['Actual']
    })
    df_plot.loc[len(df_plot) - 1, 'Type'] = 'Prediction'

    fig = px.line(df_plot, x='Day', y='Temperature', color='Type',
                  title="Temperature Predictions vs Actual",
                  labels={'Temperature': 'Temperature (°C)', 'Day': 'Day'},
                  template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Footer Section
st.markdown("""
    ---
    Made with ❤️ by ** by Boss 👦🏻 Ice 🧊 Film 🎞️**  
    """)