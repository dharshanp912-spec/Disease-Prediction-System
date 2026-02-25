import pickle

# Load Model
model = pickle.load(open("model.pkl","rb"))

print("Disease Prediction System")

# User Input
fever = int(input("Fever (0 or 1): "))
cough = int(input("Cough (0 or 1): "))
fatigue = int(input("Fatigue (0 or 1): "))

# Prediction
result = model.predict([[fever,cough,fatigue]])

if result[0] == 1:
    print("\nDisease Detected")
else:
    print("\nNo Disease Detected")