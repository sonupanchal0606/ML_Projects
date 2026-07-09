from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path

app = FastAPI()

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "house_price_model.pkl"
METRICS_PATH = PROJECT_DIR / "model" / "metrics.json"
DATA_PATH = PROJECT_DIR / "data" / "housing.csv"
STATIC_DIR = PROJECT_DIR / "app" / "static"

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

class HouseData(BaseModel):
    longitude: float
    latitude: float
    median_income: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    ocean_proximity: str

@app.get("/metrics")
def get_metrics():
    return metrics

def build_input_df(data: HouseData):
    rooms_per_household = data.total_rooms / data.households
    bedrooms_per_room = data.total_bedrooms / data.total_rooms
    population_per_household = data.population / data.households

    return pd.DataFrame([{
        "longitude": data.longitude,
        "latitude": data.latitude,
        "median_income": data.median_income,
        "housing_median_age": data.housing_median_age,
        "total_rooms": data.total_rooms,
        "total_bedrooms": data.total_bedrooms,
        "population": data.population,
        "households": data.households,
        "rooms_per_household": rooms_per_household,
        "bedrooms_per_room": bedrooms_per_room,
        "population_per_household": population_per_household,
        "ocean_proximity": data.ocean_proximity
    }])

@app.post("/predict")
def predict(data: HouseData):
    input_df = build_input_df(data)
    prediction = model.predict(input_df)

    return {
        "predicted_price": round(float(prediction[0]), 2),
        "currency": "USD",
        "model_used": metrics["model"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2_score": metrics["r2_score"],
        "note": "This is an estimated house price based on training data."
    }

@app.post("/plot-data")
def plot_data(data: HouseData):
    df = pd.read_csv(DATA_PATH)
    df = df[["median_income", "median_house_value"]].dropna()
    sample = df.sample(n=min(250, len(df)), random_state=42)

    x = df["median_income"].to_numpy()
    y = df["median_house_value"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    x_min = float(sample["median_income"].min())
    x_max = float(sample["median_income"].max())

    input_df = build_input_df(data)
    predicted_price = float(model.predict(input_df)[0])

    return {
        "points": [
            {"x": float(row.median_income), "y": float(row.median_house_value)}
            for row in sample.itertuples(index=False)
        ],
        "line": [
            {"x": x_min, "y": float((slope * x_min) + intercept)},
            {"x": x_max, "y": float((slope * x_max) + intercept)},
        ],
        "test_point": {
            "x": data.median_income,
            "y": predicted_price,
        },
    }

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
