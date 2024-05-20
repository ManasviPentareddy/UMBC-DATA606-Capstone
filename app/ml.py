# ml.py

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# Read the dataset
boston_df = pd.read_csv('Cleaned_Housing.csv')

# Extract features (X) and target (y)
X = boston_df.drop(columns=['MEDV'])
y = boston_df['MEDV']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler object
joblib.dump(scaler, 'scaler.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)

# Save the model
joblib.dump(model, 'house_price_prediction_model.pkl')
