import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(path, target_column):

    df = pd.read_csv(path)

    # Split features & target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Keep only numeric features
    X = X.select_dtypes(include=['number'])

    # Save feature columns (IMPORTANT for prediction)
    feature_columns = X.columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_columns, le