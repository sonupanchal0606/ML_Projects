import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("../data/housing.csv")

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
joblib.dump(model, "../model/house_price_model.pkl")

print("Model saved")