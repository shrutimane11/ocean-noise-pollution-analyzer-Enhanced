import joblib
import pandas as pd

# Load saved components
model = joblib.load("outputs/best_model.pkl")
scaler = joblib.load("outputs/scaler.pkl")
columns = joblib.load("outputs/columns.pkl")
label_encoder = joblib.load("outputs/label_encoder.pkl")

print("✅ Model, scaler, columns, encoder loaded")

# Example input (replace values if needed)
data = pd.DataFrame([{
    "Sensitivity (dB)": 120,
    "Recorder Depth (m)": 50,
    "Sample Rate (kHz)": 44,
    "Gain/Pre-amp": 10,
    "Bandwidth for Analysis (kHz)": 20,
    "Location Latitude": 18.5,
    "Location Longitude": 73.8,
    "Frequency Mean": 500,
    "Power (dB)": 80,
    "MFCC 1": 0.5,
    "MFCC 2": 0.3,
    "SNR": 15,
    "Background Noise": 30
}])

# Match training columns
data = data.reindex(columns=columns, fill_value=0)

# Scale
data_scaled = scaler.transform(data)

# Predict
prediction_encoded = model.predict(data_scaled)

# Convert back to original label
prediction_label = label_encoder.inverse_transform(prediction_encoded)

print("🔮 Final Prediction:", prediction_label[0])