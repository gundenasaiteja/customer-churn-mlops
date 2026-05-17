import pandas as pd
import joblib
import json
import mlflow

from pipeline import clean_data, engineer_features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Data pipeline
df = clean_data(df)
df = engineer_features(df)

# One hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save feature columns
with open("models/feature_cols.json", "w") as f:
    json.dump(list(X.columns), f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# KMeans clustering
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

kmeans.fit(X_train_scaled)

joblib.dump(
    kmeans,
    "models/kmeans.pkl"
)

# MLflow tracking
mlflow.set_experiment(
    "Customer_Churn_Prediction"
)

with mlflow.start_run():

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(f"Accuracy: {accuracy}")

    # Save model
    joblib.dump(
        model,
        "models/model.pkl"
    )

    # MLflow logs
    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "max_depth",
        10
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

print("Training completed successfully!")