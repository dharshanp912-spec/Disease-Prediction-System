import pickle
import numpy as np

# Load Saved Model
model = pickle.load(open("model.pkl", "rb"))

print("=== Heart Disease Prediction ===")

# User Input
age=int(input("Age: "))
fever=int(input("Fever (0/1): "))
cough=int(input("Cough (0/1): "))
fatigue=int(input("Fatigue (0/1): "))
chest_pain=int(input("Chest Pain (0/1): "))
blood_pressure=int(input("Blood Pressure: "))
cholesterol=int(input("Cholesterol Level: "))
sugar_level=int(input("Sugar Level: "))
# Store Input
input_data=np.array([[age, fever, cough, fatigue,
                        chest_pain, blood_pressure,
                        cholesterol, sugar_level]])
# Prediction
result=model.predict(input_data)
# Output
if result[0]==1:
    print("\nHeart Disease Detected")
else:
    print("\nNo Heart Disease")
