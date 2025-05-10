import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Define the model
rf = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Train the model with the best hyperparameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_scaled, y_train)

# Evaluate the best model
accuracy = best_rf.score(X_test_scaled, y_test)
print(f"Model accuracy with best hyperparameters: {accuracy}")

# Save the best model and scaler
with open('D:/Fetal Health 2/fetal-health-system/backend/model/fetal_best_model_713PM.pkl', 'wb') as model_file:
    pickle.dump(best_rf, model_file)

with open('D:/Fetal Health 2/fetal-health-system/backend/model/scaler_713PM.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Best model and scaler saved successfully.")
