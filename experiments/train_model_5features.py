import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Load training data
train_df = pd.read_csv('data/train_df.csv')

# Convert gender to numeric
train_df['gender_Male'] = train_df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Select 5 features
FEATURES = ['age', 'num_procedures', 'days_in_hospital', 'comorbidity_score', 'gender_Male']
X = train_df[FEATURES]
y = train_df['readmitted']

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Validation
y_val_pred = rf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Retrain on full dataset
rf.fit(X, y)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(rf, "models/rf_readmission.pkl")
print("Model saved as 'models/rf_readmission.pkl' with 5 features")
