import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("../data/housing.csv")

# Create new useful features
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# Remove rows with missing values
df = df.dropna()

# Input features
features = [
    "longitude",
    "latitude",
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "ocean_proximity"
]

X = df[features]
y = df["median_house_value"]

# Numeric and categorical columns
numeric_features = [
    "longitude",
    "latitude",
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household"
]

categorical_features = ["ocean_proximity"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Model trained successfully")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "../model/house_price_model.pkl")

# Save metrics
metrics = {
    "model": "RandomForestRegressor",
    "mae": round(mae, 2),
    "rmse": round(rmse, 2),
    "r2_score": round(r2, 4)
}

with open("../model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model and metrics saved successfully")