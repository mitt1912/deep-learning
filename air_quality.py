import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Step 1: Load the dataset
df = pd.read_csv('air_quality_dataset.csv')  # Change the filename if needed

# Display the first few rows of the dataset
print(df.head())

# Step 2: Data Preprocessing

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (you can also use imputation if necessary)
df.dropna(inplace=True)

# Check for duplicates and drop them
df.drop_duplicates(inplace=True)

# Check for correlation between features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Step 3: Define the target variable and features
X = df.drop(columns=['PM2.5'])  # Features (drop the target column)
y = df['PM2.5']  # Target (PM2.5)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {X_train.shape}")
print(f"Test data size: {X_test.shape}")

# Step 5: Train a Linear Regression Model

# Create a Linear Regression model
linear_model = LinearRegression()

# Train the model on the training data
linear_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_model.predict(X_test)

# Step 6: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression - Mean Absolute Error: {mae}")
print(f"Linear Regression - Mean Squared Error: {mse}")
print(f"Linear Regression - Root Mean Squared Error: {rmse}")
print(f"Linear Regression - R-squared: {r2}")

# Step 7: Plot Actual vs Predicted PM2.5 values for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression - Actual vs Predicted PM2.5')
plt.show()

# Step 8: Try a Random Forest Regressor for Comparison

# Create a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_rf_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_mae = mean_absolute_error(y_test, y_rf_pred)
rf_mse = mean_squared_error(y_test, y_rf_pred)
rf_rmse = mean_squared_error(y_test, y_rf_pred, squared=False)
rf_r2 = r2_score(y_test, y_rf_pred)

print(f"Random Forest - Mean Absolute Error: {rf_mae}")
print(f"Random Forest - Mean Squared Error: {rf_mse}")
print(f"Random Forest - Root Mean Squared Error: {rf_rmse}")
print(f"Random Forest - R-squared: {rf_r2}")

# Step 9: Plot Actual vs Predicted PM2.5 values for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_rf_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest - Actual vs Predicted PM2.5')
plt.show()

# Optional: Save the model for later use
import joblib
joblib.dump(rf_model, 'random_forest_air_quality_model.pkl')

# You can later load the model with:
# rf_model = joblib.load('random_forest_air_quality_model.pkl')
