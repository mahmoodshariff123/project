import pickle
from sklearn.svm import SVC
import pandas as pd

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Assuming the target column is named 'Outcome' (change it according to your CSV file)
X = data.drop('Outcome', axis=1)  # Drop the target column to get features
y = data['Outcome']  # Target column

# Train the model
model = SVC()
model.fit(X, y)

# Save the model to a file
with open('Diabetesmodel.pkl', 'wb') as file:
    pickle.dump(model, file)

