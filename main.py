import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load dataset
data = pd.read_csv('employee_data.csv')

# Data cleaning
data.dropna(inplace=True)

# Text processing for job descriptions
vectorizer = CountVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(data['JobDescription'])

# Combine numerical features with text features
X_num = data[['YearsExperience', 'EducationLevel']]
X = np.hstack((X_num, X_text.toarray()))

# Target variable
y = data['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Random Forest RMSE: {rmse:.2f}')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the neural network
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1, activation='linear'))

# Compile the model
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predictions
y_pred_nn = nn_model.predict(X_test)

# Evaluation
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
print(f'Neural Network RMSE: {rmse_nn:.2f}')


import streamlit as st

st.title('Employee Salary Prediction')

# Input fields
years_experience = st.number_input('Years of Experience', min_value=0)
education_level = st.selectbox('Education Level', ['High School', 'Bachelor', 'Master', 'PhD'])
job_description = st.text_area('Job Description')

# Predict button
if st.button('Predict Salary'):
    # Preprocess input and make prediction
    input_data = np.array([[years_experience, education_level, job_description]])
    input_vectorized = vectorizer.transform(input_data[:, 2])
    input_combined = np.hstack((input_data[:, :2], input_vectorized.toarray()))
    
    salary_prediction = rf_model.predict(input_combined)
    st.write(f'Predicted Salary: ${salary_prediction[0]:,.2f}')

