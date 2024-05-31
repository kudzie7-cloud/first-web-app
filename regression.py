import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate some example data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Convert to DataFrame for better manipulation
data = pd.DataFrame(data=np.hstack((X, y)), columns=['X', 'y'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['X']], data['y'], test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model using joblib
import joblib
joblib.dump(model, 'linear_regression_model.pkl')

import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

st.title("Simple Linear Regression Model")

st.write("""
### Predicting a Value Using Simple Linear Regression
""")

# Input feature value
input_value = st.number_input("Enter a value for X:", value=0.0)

# Prediction
if st.button("Predict"):
    prediction = model.predict(np.array([[input_value]]))[0]
    st.write(f"Predicted value for y: {prediction}")

