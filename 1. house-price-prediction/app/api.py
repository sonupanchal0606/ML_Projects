from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("../model/house_price_model.pkl")

class HouseData(BaseModel):
    median_income: float
    housing_median_age: float
    total_rooms: float
    population: float
    households: float

@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict(data: HouseData):

    input_data = np.array([[
        data.median_income,
        data.housing_median_age,
        data.total_rooms,
        data.population,
        data.households
    ]])

    prediction = model.predict(input_data)

    return {
        "predicted_price": round(float(prediction[0]), 2)
    }