import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv('D:/Fetal Health 2/fetal-health-system/data/fetal_health.csv')

# Split the data into features and labels
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the features
print("Features shape:", X.shape)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open('D:/Fetal Health 2/fetal-health-system/backend/model/fetal_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('D:/Fetal Health 2/fetal-health-system/backend/model/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"Model trained and saved successfully.")
