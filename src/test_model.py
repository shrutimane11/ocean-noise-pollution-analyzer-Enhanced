print("🔮 Testing started")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/sonar_acoustic_dataset.csv")

print("📊 Dataset loaded:", df.shape)

# Split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("🤖 Model trained for testing")

# Predictions
y_pred = model.predict(X_test)

# -------------------------
# MODEL EVALUATION
# -------------------------
model_accuracy = accuracy_score(y_test, y_pred)

print("\n✅ TEST RESULTS")
print("Accuracy:", model_accuracy)

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# BASELINE MODEL
# -------------------------
majority_class = np.bincount(y_test).argmax()
baseline_pred = np.full(len(y_test), majority_class)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("\n🎯 BASELINE RESULT")
print("Baseline Accuracy:", baseline_accuracy)

# -------------------------
# FINAL MODEL RESULT
# -------------------------
print("\n🤖 MODEL RESULT")
print("Model Accuracy:", model_accuracy)

print("\n🎯 Testing completed")

import matplotlib.pyplot as plt

# Values
labels = ["Baseline", "Model"]
values = [baseline_accuracy, model_accuracy]

# Plot
plt.bar(labels, values)

plt.title("Baseline vs Model Comparison")
plt.ylabel("Accuracy")

plt.show()