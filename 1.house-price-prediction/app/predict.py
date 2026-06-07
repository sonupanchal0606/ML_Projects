import joblib
import numpy as np

# Load model
model = joblib.load("../model/house_price_model.pkl")

# Example input
sample_house = np.array([[
    8.3252,   # median_income
    41,       # housing_median_age
    880,      # total_rooms
    322,      # population
    126       # households
]])

prediction = model.predict(sample_house)

print("Predicted House Price:")
print(prediction[0])