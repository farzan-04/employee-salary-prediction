import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("employee_data.csv")

data = load_data()

# Clean and preprocess
data.dropna(inplace=True)

# Encode education level to numeric
edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
data['EducationLevel'] = data['EducationLevel'].map(edu_map)

# Vectorize text
vectorizer = CountVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(data['JobDescription'])

# Combine features
X_num = data[['YearsExperience', 'EducationLevel']].values
X = np.hstack((X_num, X_text.toarray()))
y = data['Salary'].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Streamlit App
st.title("Employee Salary Prediction")
st.markdown(f"**Model Accuracy (Random Forest RMSE):** ${rmse:,.2f}")

# User input
st.header("Enter Employee Info")
years_experience = st.number_input("Years of Experience", min_value=0, max_value=50)
education_level = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD'])
job_description = st.text_area("Job Description")

# Predict button
if st.button("Predict Salary"):
    edu_level_num = edu_map[education_level]
    text_features = vectorizer.transform([job_description])
    combined_input = np.hstack((
        np.array([[years_experience, edu_level_num]]), 
        text_features.toarray()
    ))

    prediction = rf_model.predict([combined_input[0]])
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
