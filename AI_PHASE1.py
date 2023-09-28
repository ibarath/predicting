# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your dataset (replace 'data.csv' with your dataset's file path)
data = pd.read_csv('data.csv')

# Define your features (X) and target variable (y)
X = data[['Feature1', 'Feature2', 'Feature3']]  # Replace with your actual features
y = data['Price']  # Replace with your target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Now you can use the trained model to predict house prices for new data
new_data = pd.DataFrame({'Feature1': [value1], 'Feature2': [value2], 'Feature3': [value3]})
predicted_price = model.predict(new_data)
print(f'Predicted Price: {predicted_price[0]}')
