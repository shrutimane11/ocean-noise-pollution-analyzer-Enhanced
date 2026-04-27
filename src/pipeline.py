import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ---------------------------
# SET BASE PATH
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "sonar_acoustic_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "leaderboard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model.pkl")

# ---------------------------
# LOAD DATASET
# ---------------------------
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

# ---------------------------
# PREPROCESS
# ---------------------------
def preprocess(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

# ---------------------------
# EVALUATE
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

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(BASE_DIR, "outputs", f"{type(model).__name__}_cm.png"))
    plt.close()

    return acc, prec, rec, f1

# ---------------------------
# UPDATE LEADERBOARD
# ---------------------------
def update_leaderboard(model_name, acc, prec, rec, f1):
    new_entry = {
        "Model": model_name,
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
# PIPELINE
# ---------------------------
def run_pipeline():
    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, MODEL_PATH)

    acc, prec, rec, f1 = evaluate(rf_model, X_test, y_test)
    update_leaderboard("RandomForest", acc, prec, rec, f1)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    acc, prec, rec, f1 = evaluate(lr_model, X_test, y_test)
    update_leaderboard("LogisticRegression", acc, prec, rec, f1)

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    acc, prec, rec, f1 = evaluate(dt_model, X_test, y_test)
    update_leaderboard("DecisionTree", acc, prec, rec, f1)

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print("❌ ERROR:", e)