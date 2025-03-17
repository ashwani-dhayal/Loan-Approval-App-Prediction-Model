import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle

# Sample data creation (since we don't have the actual dataset)
def create_sample_data(n_samples=100):
    np.random.seed(42)  # For reproducibility
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], size=n_samples),
        'Married': np.random.choice(['Yes', 'No'], size=n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], size=n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], size=n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], size=n_samples),
        'ApplicantIncome': np.random.randint(1000, 10000, size=n_samples),
        'CoapplicantIncome': np.random.randint(0, 5000, size=n_samples),
        'LoanAmount': np.random.randint(50, 500, size=n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 240, 120], size=n_samples),
        'Credit_History': np.random.choice([0.0, 1.0], size=n_samples, p=[0.2, 0.8]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], size=n_samples),
        'Loan_Status': np.random.choice(['Y', 'N'], size=n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

# Load or create data
try:
    print("Attempting to load dataset...")
    loan_data = pd.read_csv('loan_prediction_dataset.csv')
    print("Dataset loaded successfully")
except FileNotFoundError:
    print("Creating sample dataset...")
    loan_data = create_sample_data(n_samples=500)
    print(f"Sample dataset created with {len(loan_data)} records")

# Data preprocessing
# Handle missing values (if any)
loan_data = loan_data.fillna(loan_data.mode().iloc[0])

# Convert categorical variables to dummy variables
loan_data = pd.get_dummies(loan_data, drop_first=True)

# Define features and target
X = loan_data.drop('Loan_Status_Y', axis=1) if 'Loan_Status_Y' in loan_data.columns else loan_data.drop('Loan_Status', axis=1)
Y = loan_data['Loan_Status_Y'] if 'Loan_Status_Y' in loan_data.columns else (loan_data['Loan_Status'] == 'Y')

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
model.fit(X_train, Y_train)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))

# Save the feature names and scaler for later use
pickle.dump({'feature_names': X.columns.tolist(), 'scaler': scaler}, open('model_metadata.pkl', 'wb'))

# Evaluate the model
train_accuracy = model.score(X_train, Y_train)
test_accuracy = model.score(X_test, Y_test)

print(f"Model trained successfully!")
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Testing accuracy: {test_accuracy*100:.2f}%")
