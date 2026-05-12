import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px

from src.pipeline import clean_data, engineer_features


# Load models
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")

with open("models/feature_cols.json", "r") as f:
    feature_cols = json.load(f)


st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")


# Sidebar
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Overview",
        "Predict Churn",
        "Customer Segments"
    ]
)


# Load data
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df = clean_data(df)
df = engineer_features(df)


# OVERVIEW PAGE
if page == "Overview":

    st.header("Overview Dashboard")

    total_customers = len(df)
    churn_rate = df["Churn"].mean() * 100

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Customers",
        total_customers
    )

    col2.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    churn_counts = df["Churn"].value_counts()

    fig = px.pie(
        values=churn_counts.values,
        names=["Churned", "Retained"],
        title="Customer Churn Distribution"
    )

    st.plotly_chart(fig)


# PREDICTION PAGE
elif page == "Predict Churn":

    st.header("Customer Churn Prediction")

    tenure = st.slider(
        "Tenure",
        1,
        72,
        12
    )

    monthly = st.slider(
        "Monthly Charges",
        10,
        150,
        70
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    if st.button("Predict"):

        sample = {
            "gender": 1,
            "SeniorCitizen": 0,
            "Partner": 1,
            "Dependents": 0,
            "tenure": tenure,
            "PhoneService": 1,
            "MultipleLines": "No",
            "InternetService": internet,
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": 1,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
            "Churn": 0
        }

        sample_df = pd.DataFrame([sample])

        sample_df = clean_data(sample_df)
        sample_df = engineer_features(sample_df)

        sample_df = pd.get_dummies(
            sample_df
        )

        for col in feature_cols:
            if col not in sample_df.columns:
                sample_df[col] = 0

        sample_df = sample_df[feature_cols]

        scaled = scaler.transform(sample_df)

        probability = model.predict_proba(
            scaled
        )[0][1]

        st.subheader(
            f"Churn Probability: {probability*100:.2f}%"
        )

        if probability > 0.65:
            st.error("High Churn Risk")

        elif probability > 0.35:
            st.warning("Medium Churn Risk")

        else:
            st.success("Low Churn Risk")


# CLUSTER PAGE
elif page == "Customer Segments":

    st.header("Customer Segments")

    temp = pd.get_dummies(
        df.drop("Churn", axis=1),
        drop_first=True
    )

    scaled = scaler.transform(temp)

    clusters = kmeans.predict(scaled)

    temp["Cluster"] = clusters

    fig = px.scatter(
        temp,
        x="tenure",
        y="MonthlyCharges",
        color="Cluster",
        title="Customer Segments"
    )

    st.plotly_chart(fig)