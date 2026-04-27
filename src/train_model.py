import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("🚀 Training started")

import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("data/sonar_acoustic_dataset.csv")
print("📊 Columns in dataset:", df.columns)

TARGET_COLUMN = "Sound Source"

# Import preprocessing
from src.preprocess import preprocess_data

print("📊 Preprocessing data...")

X_train, X_test, y_train, y_test, scaler, feature_columns, label_encoder = preprocess_data(
    "data/sonar_acoustic_dataset.csv",
    target_column=TARGET_COLUMN
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

results = {}

print("🔥 Training models...")

for name, model in models.items():
    print(f"\n🤖 Training {name}...")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    results[name] = acc
    print(f"✅ {name} Accuracy: {acc:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds))

    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

print("\n🏆 FINAL RESULTS:")
print(results)

# Leaderboard
from src.leaderboard import initialize_leaderboard, add_entry

initialize_leaderboard()

for name, acc in results.items():
    add_entry(name, acc, acc, acc, acc)

# Best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"🏆 Best Accuracy: {results[best_model_name]:.4f}")

# Save everything
os.makedirs("outputs", exist_ok=True)

joblib.dump(best_model, "outputs/best_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
joblib.dump(feature_columns, "outputs/columns.pkl")
joblib.dump(label_encoder, "outputs/label_encoder.pkl")

print("💾 Model, scaler, columns, encoder saved!")

print("🎯 Training completed successfully")