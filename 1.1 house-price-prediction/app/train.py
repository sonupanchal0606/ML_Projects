import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "housing.csv"
MODEL_PATH = PROJECT_DIR / "model" / "house_price_model.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Select features
X = df[[
    "median_income",
    "housing_median_age",
    "total_rooms",
    "population",
    "households"
]]

# Target
y = df["median_house_value"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)

print("Model trained successfully")
print(f"Mean Absolute Error: {mae}")

# Save model
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Model saved")
