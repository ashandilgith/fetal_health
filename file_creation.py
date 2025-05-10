import os

# Define the directory structure
directories = [
    "fetal-health-system/backend",
    "fetal-health-system/backend/model",
    "fetal-health-system/backend/utils",
    "fetal-health-system/data"
]

# Define the files to be created
files = [
    "fetal-health-system/backend/app.py",              # Main API application (Flask)
    "fetal-health-system/backend/model/model.py",      # Model creation and training
    "fetal-health-system/backend/model/fetal_model.pkl", # Saved model after training
    "fetal-health-system/backend/utils/alerts.py",     # Alerting system (email, phone)
    "fetal-health-system/backend/requirements.txt",     # Dependencies
    "fetal-health-system/data/fetal_health.csv",       # Dataset from Kaggle
    "fetal-health-system/README.md"                     # Documentation
]

# Create directories
for directory in directories:
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")

# Create empty files
for file in files:
    try:
        with open(file, 'w') as f:
            f.write("")  # Creating empty files
        print(f"File created: {file}")
    except Exception as e:
        print(f"Error creating file {file}: {e}")
