# 🏠 House Price Prediction using Machine Learning

A Machine Learning project that predicts house prices using housing-related features such as income, house age, number of rooms, population, and households.

This project demonstrates a complete ML workflow:

* Data Loading
* Data Preprocessing
* Model Training
* Model Evaluation
* Model Persistence
* Prediction API using FastAPI
* Swagger API Documentation

---

# 📌 Features

* Train a Linear Regression model
* Predict house prices from user input
* Save and load trained models
* REST API using FastAPI
* Interactive Swagger UI
* Beginner-friendly project structure
* Easily extendable with Random Forest and XGBoost

---

# 🛠 Tech Stack

### Machine Learning

* Python 3.10+
* Pandas
* NumPy
* Scikit-Learn
* Joblib

### API

* FastAPI
* Uvicorn

### Visualization

* Matplotlib

---

# 📂 Project Structure

```text
house-price-prediction/
│
├── data/
│   └── housing.csv
│
├── model/
│   └── house_price_model.pkl
│
├── app/
│   ├── train.py
│   ├── predict.py
│   ├── api.py
│   └── requirements.txt
│
├── README.md
│
└── venv/
```

---

# 🚀 Getting Started

## Prerequisites

Verify Python installation:

```bash
python3 --version
```

Expected:

```text
Python 3.x.x
```

---

# Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/house-price-prediction.git

cd house-price-prediction
```

---

# Step 2: Create Virtual Environment

Mac/Linux

```bash
python3 -m venv venv
```

Activate environment:

```bash
source venv/bin/activate
```

Expected:

```text
(venv) username@MacBook-Air
```

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

# Step 3: Install Dependencies

Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib joblib fastapi uvicorn
```

Save requirements:

```bash
pip freeze > app/requirements.txt
```

OR

Install directly from requirements:

```bash
pip install -r app/requirements.txt
```

---

# Step 4: Dataset Setup

Download California Housing Dataset.

Recommended source:

https://www.kaggle.com/datasets/camnugent/california-housing-prices

Place dataset here:

```text
data/housing.csv
```

---

# Dataset Columns

Input Features

```text
median_income
housing_median_age
total_rooms
population
households
```

Target

```text
median_house_value
```

---

# Step 5: Train Model

Navigate to app folder:

```bash
cd app
```

Run training:

```bash
python3 train.py
```

Expected Output

```text
Model trained successfully
Mean Absolute Error: 58230.45
Model saved
```

Generated file:

```text
model/house_price_model.pkl
```

---

# Training Workflow

```text
Load Dataset
      ↓
Feature Selection
      ↓
Train/Test Split
      ↓
Train Model
      ↓
Evaluate Model
      ↓
Save Model
```

---

# Step 6: Test Prediction Script

Run:

```bash
python3 predict.py
```

Expected Output

```text
Predicted House Price:
412345.23
```

---

# Step 7: Run FastAPI Server

Start API server:

```bash
uvicorn api:app --reload
```

Expected:

```text
INFO: Uvicorn running on http://127.0.0.1:8000
```

---

# Step 8: Swagger Documentation

Open browser:

```text
http://127.0.0.1:8000/docs
```

Swagger UI will automatically generate API documentation.

---

# API Endpoints

---

## Health Check

### Request

```http
GET /
```

### Response

```json
{
  "message": "House Price Prediction API"
}
```

---

## Predict House Price

### Request

```http
POST /predict
```

### Request Body

```json
{
  "median_income": 8.3252,
  "housing_median_age": 41,
  "total_rooms": 880,
  "population": 322,
  "households": 126
}
```

### Success Response

```json
{
  "predicted_price": 412345.23
}
```

---

# Testing with Swagger

1. Open:

```text
http://127.0.0.1:8000/docs
```

2. Expand:

```text
POST /predict
```

3. Click:

```text
Try it out
```

4. Enter request JSON

5. Click:

```text
Execute
```

6. View response

---

# Testing with cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
"median_income":8.3252,
"housing_median_age":41,
"total_rooms":880,
"population":322,
"households":126
}'
```

---

# Sample Prediction Flow

```text
User Input
     ↓
API Request
     ↓
Load Trained Model
     ↓
Model Prediction
     ↓
Return Predicted Price
```

---

# Evaluation Metric

This project uses:

### Mean Absolute Error (MAE)

Formula:

MAE = Average(|Actual - Predicted|)

Lower MAE indicates better model performance.

---

# Current Model

```text
Linear Regression
```

Advantages:

* Simple
* Fast
* Easy to understand
* Great for beginners

---

# Future Improvements

## Machine Learning

* Feature Scaling
* Cross Validation
* Hyperparameter Tuning
* Feature Engineering

## Models

* Random Forest Regressor
* XGBoost Regressor
* Gradient Boosting Regressor

## Deep Learning

* ANN Regression Model
* TensorFlow/Keras implementation

## Deployment

* Docker
* AWS EC2
* Azure App Service
* Render

## Frontend

* React
* Next.js
* ASP.NET MVC

---

# Learning Outcomes

By completing this project you will learn:

* Supervised Learning
* Regression Problems
* Data Preprocessing
* Train/Test Split
* Model Training
* Model Evaluation
* Model Persistence
* REST API Development
* API Testing
* Production ML Workflow

---

# Next Projects

After completing this project:

1. Spam Email Classifier
2. Customer Churn Prediction
3. Sentiment Analysis
4. MNIST ANN
5. CNN Image Classifier
6. Stock Trend Prediction
7. Resume Screening System
8. PDF Chatbot (RAG)
9. News + Stock Analysis Assistant

---

# Author

Sonu Panchal

Backend Developer 
