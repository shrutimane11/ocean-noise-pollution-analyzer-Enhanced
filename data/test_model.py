print("🔮 Testing started")

import pandas as pd
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

# Model (same as training file for now)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("🤖 Model trained for testing")

# Testing
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)

print("\n✅ TEST RESULTS")
print("Accuracy:", acc)

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

print("🎯 Testing completed")