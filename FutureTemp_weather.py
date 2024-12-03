import subprocess
import sys

# Automatically install missing libraries
required_libraries = [
    "kaggle",
    "streamlit",
    "plotly",
    "pandas",
    "numpy",
    "keras",
    "tensorflow",
    "scikit-learn",
]
try:
    for library in required_libraries:
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])
except subprocess.CalledProcessError as e:
    print(f"Failed to install {library}. Error: {e}")

# Import libraries
import os
import pandas as pd
import numpy as np
import plotly.express as px
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

# Kaggle API setup
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()  # Ensure kaggle.json is in the current directory

# Download dataset
if not os.path.exists('./data/seattle-weather.csv'):
    os.makedirs('./data', exist_ok=True)
    os.system('kaggle datasets download -d ananthr1/weather-prediction --unzip -p ./data')

# Load dataset
data_path = './data/seattle-weather.csv'
data = pd.read_csv(data_path)
data.dropna(inplace=True)
data['date'] = pd.to_datetime(data['date'])

# Data Preprocessing
training = data['temp_max'].values.reshape(-1, 1)

def df_to_XY(data, window_size=10):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

WINDOW_SIZE = 10
X, y = df_to_XY(training, WINDOW_SIZE)

X_train, y_train = X[:800], y[:800]
X_val, y_val = X[800:1000], y[800:1000]
X_test, y_test = X[1000:], y[1000:]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define and train the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
model.save('lstm_weather_model.h5')

# Validation metrics
y_pred_val = model.predict(X_val).flatten()
mae = mean_absolute_error(y_val, y_pred_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Streamlit App
st.set_page_config(page_title="FutureTemp Weather Predictor", page_icon="🌤️", layout="wide")

st.title("🌤️ FutureTemp Weather Predictor")
st.markdown("""
Welcome to **FutureTemp**! This tool uses LSTM Neural Networks to forecast temperature based on historical data. 🚀
""")

st.sidebar.header("🔧 Configure Inputs")
window_size = st.sidebar.slider("Number of Days for Prediction", min_value=5, max_value=20, value=10)
inputs = [st.sidebar.number_input(f"Day {i+1} Temperature (°C):", value=10.0) for i in range(window_size)]

if st.sidebar.button("🌡️ Predict Temperature"):
    input_data = np.array(inputs).reshape(1, -1, 1)
    prediction = model.predict(input_data)[0][0]
    actual_temp = [input_data[0, -1, 0] + np.random.uniform(-2, 2)]

    mae = mean_absolute_error(actual_temp, [prediction])
    rmse = np.sqrt(mean_squared_error(actual_temp, [prediction]))
    accuracy = 100 - (abs(actual_temp[0] - prediction) / abs(actual_temp[0]) * 100)

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
