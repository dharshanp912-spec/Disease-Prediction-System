import pandas as pd
import sklearn.model_selection
import skl3earn.ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load Dataset
data = pd.read_csv("dataset.csv")

print("Dataset Preview:")
print(data.head())

# Features
X = data[['age','fever','cough','fatigue',
          'chest_pain','blood_pressure',
          'cholesterol','sugar_level']]

# Target
y = data['heart_disease']

# Split Dataset
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy =", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel Saved Successfully")

# User Input
print("\nEnter Patient Details")

age = int(input("Age: "))
fever = int(input("Fever (0/1): "))
cough = int(input("Cough (0/1): "))
fatigue = int(input("Fatigue (0/1): "))
chest_pain = int(input("Chest Pain (0/1): "))
blood_pressure = int(input("Blood Pressure: "))
cholesterol = int(input("Cholesterol Level: "))
sugar_level = int(input("Sugar Level: "))

# Prediction
result = model.predict([[age, fever, cough, fatigue,
                         chest_pain, blood_pressure,
                         cholesterol, sugar_level]])

# Output
if result[0] == 1:
    print("\nHeart Disease Detected")
else:
    print("\nNo Heart Disease")
