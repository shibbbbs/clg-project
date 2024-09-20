import pandas as pd
import pickle as pk
import streamlit as st

# Load pre-trained model
model = pk.load(open('treemodel.pkl', 'rb'))

# Streamlit header
st.header('Obesity Prediction')

# UI components for input
Age = st.number_input('Age', min_value=1, max_value=100, value=25)
Gender = st.selectbox('Gender', ['Male', 'Female'])  # Assuming the model expects encoded values for Gender
Height = st.number_input('Height (cm)', min_value=100, max_value=250, value=170)
Weight = st.number_input('Weight (kg)', min_value=20, max_value=300, value=70)
PhysicalActivityLevel = st.selectbox('Physical Activity Level', ['Sedentary', 'Light', 'Moderate', 'Intense'])

# Calculate BMI based on height and weight
BMI = Weight / ((Height / 100) ** 2)  # BMI formula: weight(kg) / (height(m))^2

# Display the calculated BMI
st.markdown(f'Calculated BMI: {BMI:.2f}')

# Prediction button
if st.button('Predict'):
    # Encoding Gender and Physical Activity Level
    Gender = 1 if Gender == 'Male' else 0  # Example encoding: Male -> 1, Female -> 0
    activity_mapping = {'Sedentary': 1, 'Light': 2, 'Moderate': 3, 'Intense': 4}
    PhysicalActivityLevel = activity_mapping[PhysicalActivityLevel]
    
    # Create input data for the model with the correct feature names
    input_data = pd.DataFrame([[Age, Gender, Height, Weight, BMI, PhysicalActivityLevel]],
                              columns=["Age", "Gender", "Height", "Weight", "BMI", "PhysicalActivityLevel"])  # Ensure these match model training features

    # Predict the obesity category
    ObesityCategory = model.predict(input_data)

    # Map numerical predictions to descriptive categories
    obesity_category_map = {0: 'Normal Weight', 1: 'Obese', 2: 'OverWeight', 3: 'Underweight'}
    predicted_label = obesity_category_map.get(int(ObesityCategory[0]), 'Unknown')

    # Display the result
    st.markdown(f'The predicted Obesity Category is: {predicted_label}')
