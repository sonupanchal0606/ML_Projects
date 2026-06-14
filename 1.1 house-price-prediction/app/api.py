from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "housing.csv"
MODEL_PATH = PROJECT_DIR / "model" / "house_price_model.pkl"
STATIC_DIR = BASE_DIR / "static"
FEATURE_COLUMNS = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "population",
    "households",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Main prediction model saved by train.py.
# This model uses all columns in FEATURE_COLUMNS to predict the house price.
model = joblib.load(MODEL_PATH)

housing_df = pd.read_csv(DATA_PATH)

# Data used only for the 2D graph.
# The graph shows median_income on the x-axis and median_house_value on the y-axis.
plot_df = housing_df[["median_income", "median_house_value"]].dropna()

# Separate simple model used only to draw the green regression line on the graph.
# This does not control the real prediction result returned by /predict.
plot_model = LinearRegression()
plot_model.fit(plot_df[["median_income"]], plot_df["median_house_value"])

# Keep the graph lightweight by plotting up to 500 sample rows instead of every row.
plot_sample = plot_df.sample(
    n=min(500, len(plot_df)),
    random_state=42,
).sort_values("median_income")

# Build two points for the green regression line: one at the lowest income
# and one at the highest income in the dataset.
min_income = float(plot_df["median_income"].min())
max_income = float(plot_df["median_income"].max())
line_inputs = pd.DataFrame(
    {"median_income": [min_income, max_income]}
)
line_predictions = plot_model.predict(line_inputs)

class HouseData(BaseModel):
    median_income: float
    housing_median_age: float
    total_rooms: float
    population: float
    households: float

def build_input_dataframe(data: HouseData):
    return pd.DataFrame(
        [[
            data.median_income,
            data.housing_median_age,
            data.total_rooms,
            data.population,
            data.households,
        ]],
        columns=FEATURE_COLUMNS,
    )

@app.post("/predict")
def predict(data: HouseData):
    input_data = build_input_dataframe(data)
    prediction = model.predict(input_data)

    return {
        "predicted_price": round(float(prediction[0]), 2)
    }

@app.post("/plot-data")
def plot_data(data: HouseData):
    input_data = build_input_dataframe(data)

    # The red testing point uses the main model prediction, not plot_model.
    prediction = model.predict(input_data)

    return {
        "points": [
            {
                "x": round(float(row.median_income), 4),
                "y": round(float(row.median_house_value), 2),
            }
            for row in plot_sample.itertuples()
        ],
        "line": [
            {
                "x": round(min_income, 4),
                "y": round(float(line_predictions[0]), 2),
            },
            {
                "x": round(max_income, 4),
                "y": round(float(line_predictions[1]), 2),
            },
        ],
        "test_point": {
            "x": round(float(data.median_income), 4),
            "y": round(float(prediction[0]), 2),
        },
    }

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

