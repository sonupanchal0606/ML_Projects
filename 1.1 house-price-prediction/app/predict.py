import joblib
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "house_price_model.pkl"
FEATURE_COLUMNS = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "population",
    "households",
]

# Load model
model = joblib.load(MODEL_PATH)

# Example input
sample_house = pd.DataFrame(
    [[
        8.3252,  # median_income
        41,      # housing_median_age
        880,     # total_rooms
        322,     # population
        126,     # households
    ]],
    columns=FEATURE_COLUMNS,
)

prediction = model.predict(sample_house)

print("Predicted House Price:")
print(prediction[0])
