import pandas as pd
import os

FILE_PATH = "leaderboard.csv"

COLUMNS = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]


def initialize_leaderboard():
    """Create leaderboard if not exists"""
    if not os.path.exists(FILE_PATH):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(FILE_PATH, index=False)
        print("Leaderboard created.")
    else:
        print("Leaderboard already exists.")


def add_entry(model_name, accuracy, precision, recall, f1_score):
    """Add new model results"""

    df = pd.read_csv(FILE_PATH)

    new_row = {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1_score, 4)
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df = df.sort_values(by="Accuracy", ascending=False)

    df.to_csv(FILE_PATH, index=False)

    print(f"Added {model_name} to leaderboard")