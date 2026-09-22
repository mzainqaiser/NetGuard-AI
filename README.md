# NetGuard AI

### Intelligent Network Monitoring & Threat Detection System

NetGuard AI is an AI/ML-powered network security analytics project designed to analyze network traffic data, detect unusual activity, classify potential threats, and present security insights through an interactive dashboard.

## Project Overview

Modern networks generate large volumes of traffic and security-related data. Manually analyzing this data can make it difficult to identify suspicious patterns and prioritize potentially harmful events.

NetGuard AI aims to use machine learning and data analytics to transform network traffic data into understandable security insights.

The initial version will work with network traffic datasets such as CIC-IDS2017, while the architecture is designed with future real-time network monitoring in mind.

## Key Features

- Network traffic data ingestion
- Data validation and preprocessing
- Exploratory data analysis
- Machine learning-based threat classification
- Anomaly detection
- Risk assessment
- Threat categorization
- Interactive security dashboard
- Event investigation
- Traffic and threat analytics
- Model evaluation and explainability
- Security reporting

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- Jupyter Notebook
- VS Code
- Git & GitHub

## Machine Learning

The project will investigate supervised classification models and anomaly detection techniques.

Potential classification models include:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Anomaly detection will initially investigate techniques such as:

- Isolation Forest

Final model selection will be based on experimental evaluation rather than assumptions.

## Dataset

The primary dataset selected for the project is **CIC-IDS2017**, developed by the Canadian Institute for Cybersecurity at the University of New Brunswick.

The dataset contains benign and attack network traffic with a wide range of network flow features.

Dataset source:

https://www.unb.ca/cic/datasets/ids-2017.html

Raw datasets will not be committed to this repository.

## Project Architecture

```text
Network Traffic Data
        ↓
Data Ingestion
        ↓
Data Validation
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Machine Learning
   ↙           ↘
Classification  Anomaly Detection
   ↘           ↙
   Threat Analysis
        ↓
    Risk Assessment
        ↓
   Security Analytics
        ↓
   Streamlit Dashboard