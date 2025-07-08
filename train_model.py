import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
# Replace 'lung_cancer_dataset.csv' with the actual path to your dataset
data = pd.read_csv('lung_cancer_dataset.csv')

# Data preprocessing
# Encode categorical variables
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Ensure all columns except AGE and LUNG_CANCER are in [1, 2]
binary_columns = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE',
                 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
                 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
for col in binary_columns:
    data[col] = data[col].astype(int)
    if not data[col].isin([1, 2]).all():
        raise ValueError(f"Column {col} contains values other than 1 or 2")

# Check for missing values
if data.isnull().sum().any():
    data = data.dropna()
    print("Dropped rows with missing values")

# Define features and target
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Check class distribution
print("Class distribution before SMOTE:", y.value_counts().to_dict())

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Class distribution after SMOTE:", y_resampled.value_counts().to_dict())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and hyperparameter grids
models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42),
        'param_grid': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(),
        'param_grid': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'param_grid': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
    },
    'Support Vector Machine': {
        'model': SVC(random_state=42, probability=True),
        'param_grid': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'param_grid': {}  # No hyperparameters to tune for Naive Bayes
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
    }
}

# Train and evaluate models
results = {}
for name, config in models.items():
    print(f"\nTraining {name}...")
    model = config['model']
    param_grid = config['param_grid']
    
    # Perform GridSearchCV
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
        if name in ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine']:
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
        else:
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    else:
        best_model = model
        if name in ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine']:
            best_model.fit(X_train_scaled, y_train)
        else:
            best_model.fit(X_train, y_train)
    
    # Evaluate model
    if name in ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine']:
        y_pred = best_model.predict(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train_scaled if name in ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine'] else X_train, y_train, cv=5, scoring='balanced_accuracy')
    print(f"{name} Cross-Validation Balanced Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save the model
    joblib.dump(best_model, f"{name.lower().replace(' ', '_')}_model.pkl")

# Save scaler and feature columns
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("\nModel training completed. All models saved with their respective names.")