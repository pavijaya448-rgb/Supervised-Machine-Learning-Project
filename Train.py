import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r"C:\Users\WELCOME\Downloads\Telco_Customer_Churn.csv")
data

print("Shape of Data :", data.shape)

print("\n Columns of Data :\n", data.columns)

print("\n Information of Data :\n")
data.info()

print("Data Types :\n")
print(data.dtypes)

print("Missing values :\n")
print(data.isnull().sum())

data.head()

data.tail()

# Dropping Duplicates
data.drop_duplicates()

# Dropping unimportant features
drop_cols = ['customerID',
             'SeniorCitizen',
             'Partner',
             'Dependents',
             'PhoneService',
             'OnlineBackup',
             'DeviceProtection',
             'StreamingTV',
             'StreamingMovies',
             'PaperlessBilling']

data.drop(drop_cols, axis=1, inplace=True)

# Changing appropriate Datatypes
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

cat_columns = ['gender',
               'MultipleLines',
               'OnlineSecurity',
               'Contract',
               'InternetService',
               'TechSupport',
               'PaymentMethod',
               'Churn']

data[cat_columns] = data[cat_columns].astype('category')

data.dtypes

# Checking Null values
print(data.isnull().sum())

# Filling null values
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Rechecking null values
print("Missing values :",data.isnull().sum().sum())

# Data Description
print("Data Description :")
data.describe()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
binary_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'Contract', 'TechSupport', 'PaymentMethod', 'Churn']

for col in binary_cols:
    data[col] = label_encoder.fit_transform(data[col])
print(data.head())

x = data.drop(columns='Churn', axis=1)
y = data['Churn']

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)

standardized_data = scaler.transform(x)

x = standardized_data
y = data['Churn']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(" Shape of x :", x.shape , "\n Shape of x_train :", x_train.shape ,"\n Shape of x_test :", x_test.shape)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

svc = SVC()

# Model Training
svc.fit(x_train, y_train)

# Prediction
y_pred_svc = svc.predict(x_test)

# Model Evaluation
print("\n SVC Accuracy Score:", accuracy_score(y_test, y_pred_svc))
print("\n SVC Classification Report:\n", classification_report(y_test, y_pred_svc))
print("\n SVC Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

print("\n GridSearchCV for Best Model \n")

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5,  n_jobs=-1, scoring='accuracy', verbose=4)
grid_search.fit(x_train, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)

print("\n Best Parameters:", grid_search.best_params_)
print("\n Best Cross-Validation Accuracy:", grid_search.best_score_)

# Prediction
y_pred_svm = best_model.predict(x_test)

# Model Evaluation
print("\n SVM Best Test Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\n SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("\n SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

import joblib

joblib.dump(best_model,"best_model.pkl")
joblib.dump(label_encoder, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

loaded_model = joblib.load("best_model.pkl")
loaded_label_encoder = joblib.load("label_encoders.pkl")
loaded_label_encoder = joblib.load("scaler.pkl")