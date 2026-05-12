import pandas as pd
import numpy as np


def clean_data(df):

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop customerID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Encode binary columns
    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn"
    ]

    for col in binary_cols:
        df[col] = df[col].map({
            "Yes": 1,
            "No": 0
        })

    # Gender encoding
    df["gender"] = df["gender"].map({
        "Female": 1,
        "Male": 0
    })

    return df


def engineer_features(df):

    # Tenure group
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 100],
        labels=[
            "0-1 Year",
            "1-2 Years",
            "2-4 Years",
            "4-5 Years",
            "5+ Years"
        ]
    )

    # Number of services
    service_cols = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["num_services"] = 0

    for col in service_cols:
        df["num_services"] += (
            (df[col] == "Yes") |
            (df[col] == 1)
        ).astype(int)

    # Long term customer
    df["is_longterm"] = (
        df["tenure"] > 24
    ).astype(int)

    # Support feature
    df["has_support"] = (
        (df["OnlineSecurity"] == "Yes") |
        (df["TechSupport"] == "Yes")
    ).astype(int)

    # Charges per month
    df["charges_per_month"] = (
        df["TotalCharges"] /
        np.maximum(df["tenure"], 1)
    )

    # Monthly charge bins
    df["monthly_charges_bin"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 30, 60, 90, 200],
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

    # Contract numeric
    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    df["contract_numeric"] = (
        df["Contract"].map(contract_map)
    )

    # Interaction feature
    df["tenure_contract_interaction"] = (
        df["tenure"] *
        df["contract_numeric"]
    )

    return df