import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------------------------
# SET BASE PATH
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "sonar_acoustic_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "leaderboard.csv")

# ---------------------------
# LOAD DATASET
# ---------------------------
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df

# ---------------------------
# PREPROCESS DATA
# ---------------------------
def preprocess(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

# ---------------------------
# TRAIN MODEL
# ---------------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# ---------------------------
# EVALUATE MODEL
# ---------------------------
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted')
    rec = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print("\n📊 MODEL PERFORMANCE")
    print(f"Accuracy : {acc}")
    print(f"Precision: {prec}")
    print(f"Recall   : {rec}")
    print(f"F1 Score : {f1}")

    # ✅ CONFUSION MATRIX (THIS WAS MISSING)
    cm = confusion_matrix(y_test, predictions)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save to outputs folder
    plt.savefig(os.path.join(BASE_DIR, "outputs", "confusion_matrix.png"))
    plt.close()

    return acc, prec, rec, f1

# ---------------------------
# UPDATE LEADERBOARD
# ---------------------------
def update_leaderboard(acc, prec, rec, f1):
    new_entry = {
        "Model": "RandomForest",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if not os.path.exists(OUTPUT_PATH):
        df = pd.DataFrame([new_entry])
    else:
        df = pd.read_csv(OUTPUT_PATH)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    df.to_csv(OUTPUT_PATH, index=False)

# ---------------------------
# FULL PIPELINE
# ---------------------------
def run_pipeline():
    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    acc, prec, rec, f1 = evaluate(model, X_test, y_test)

    update_leaderboard(acc, prec, rec, f1)

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print("❌ ERROR:", e)