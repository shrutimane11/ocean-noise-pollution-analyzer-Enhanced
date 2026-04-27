print("🚀 Training started")

import pandas as pd

print("📊 Loading dataset...")

df = pd.read_csv("data/sonar_acoustic_dataset.csv")

print("📊 Dataset shape:", df.shape)

print("⚙ Splitting data...")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("🤖 Importing model...")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🔥 Training model...")

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("🔮 Predicting...")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("✅ Accuracy:", acc)
print("🎯 Training completed successfully")