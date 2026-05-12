# Customer Churn Prediction MLOps Project

## Overview
This project predicts telecom customer churn using Machine Learning and demonstrates an end-to-end MLOps workflow.

## Technologies Used
- Python
- Scikit-learn
- Streamlit
- MLflow
- Docker
- Kubernetes
- GitHub Actions

## Features
- Customer churn prediction
- Customer segmentation using KMeans
- Interactive dashboard
- Containerized deployment
- Kubernetes orchestration

## ML Models
- Random Forest Classifier
- KMeans Clustering

## Run Project

### Train Model
python src/train.py

### Run Streamlit
python -m streamlit run app.py

### Docker
docker build -t churn-app .
docker run -p 8501:8501 churn-app

### Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml