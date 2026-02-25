import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load Dataset
data = pd.read_csv("dataset.csv")

print("Dataset Preview:")
print(data.head())

# Features (Symptoms)
X = data[['fever','cough','fatigue']]

# Target (Disease)
y = data['heart_disease']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test,y_pred)

print("\nAccuracy =",accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save Model
pickle.dump(model,open("model.pkl","wb"))

print("\nModel Saved Successfully")