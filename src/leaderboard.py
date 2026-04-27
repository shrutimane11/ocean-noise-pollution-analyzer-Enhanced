import pandas as pd
import os

FILE_PATH = "leaderboard.csv"

COLUMNS = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]


def initialize_leaderboard():
    """Create leaderboard if it doesn't exist"""
    if not os.path.exists(FILE_PATH):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(FILE_PATH, index=False)
        print("🏆 Leaderboard created.")
    else:
        print("🏆 Leaderboard already exists.")


def add_entry(model_name, accuracy, precision, recall, f1_score):
    """Add model performance to leaderboard"""

    # Create leaderboard if missing
    if not os.path.exists(FILE_PATH):
        initialize_leaderboard()

    df = pd.read_csv(FILE_PATH)

    new_row = {
        "Model": model_name,
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1 Score": round(float(f1_score), 4)
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Sort by best accuracy
    df = df.sort_values(by="Accuracy", ascending=False)

    df.to_csv(FILE_PATH, index=False)

    print(f"✅ Added {model_name} to leaderboard")