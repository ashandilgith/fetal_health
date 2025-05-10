import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('D:/Fetal Health 2/fetal-health-system/data/fetal_health.csv')

# Split the data into features and labels
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the model
with open('D:/Fetal Health 2/fetal-health-system/backend/model/fetal_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)




# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.2f}")
