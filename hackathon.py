import streamlit as st
import joblib
import numpy as np
import os
print('Models Inported Successfully')

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model
model_path = os.path.join(script_dir, "3mtt_random_forest.joblib")
model = joblib.load(model_path)

# Load the scaler
scaler_path = os.path.join(script_dir, "scaler_3mtt.joblib")
scaler = joblib.load(scaler_path)


# Streamlit App
st.title("Diabetes Prediction Model")
st.write("This model predicts the likelihood of having diabetes based on user inputs.")
print('Stream App Created')

# User inputs for prediction
st.header("Enter your information:")
print("Header Input Created")

# Feature inputs
high_bp = st.selectbox("Do you have high blood pressure? (1: Yes, 0: No)", options=[1, 0])
high_chol = st.selectbox("Do you have high cholesterol? (1: Yes, 0: No)", options=[1, 0])
chol_check = st.selectbox("Have you checked your cholesterol in the last 5 years? (1: Yes, 0: No)", options=[1, 0])
bmi = st.number_input("Body Mass Index (BMI)", value=25.0, step=0.1)
smoker = st.selectbox("Are you a smoker? (1: Yes, 0: No)", options=[1, 0])
stroke = st.selectbox("Any History of stroke? (1: Yes, 0: No)", options=[1, 0])
heart_disease = st.selectbox("Any History of heart disease or attack? (1: Yes, 0: No)", options=[1, 0])
PhysActivity = st.selectbox("Any Physical activity in past 30 days - not including job(0 = No 1 = Yes)", options =[1,0])
fruits = st.selectbox("Do you consume fruits frequently? (1: Yes, 0: No)", options=[1, 0])
veggies = st.selectbox("Do you consume vegetables frequently? (1: Yes, 0: No)", options=[1, 0])
heavy_alcohol = st.selectbox("Do you consume alcohol heavily? (1: Yes, 0: No)", options=[1, 0])
any_healthcare = st.selectbox("Do you have access to healthcare? (1: Yes, 0: No)", options=[1, 0])
no_docbc_cost = st.selectbox("Couldn’t see a doctor due to cost? (1: Yes, 0: No)", options=[1, 0])
gen_health = st.selectbox("Rate your general health (1: Excellent, ..., 5: Poor)", options=[1, 2, 3, 4, 5])
mental_health = st.number_input("Number of days of poor mental health in the past month", value=0, step=1)
physical_health = st.number_input("Number of days of poor physical health in the past month", value=0, step=1)
diff_walk = st.selectbox("Difficulty walking or climbing stairs? (1: Yes, 0: No)", options=[1, 0])
sex = st.selectbox("Sex (1: Male, 0: Female)", options=[1, 0])
age = st.selectbox("Age (1: 18–24, 2: 25–29, 3: 30–34, 4: 35–39, 5: 40–44, 6: 45–49, 7: 50–54, 8: 55–59, 9: 60–64, 10: 65–69, 11: 70–74, 12: 75–79, 13: 80+)", options=list(range(1, 14)))
education = st.selectbox("Education level (1: No high school, 2: High school graduate, 3: Some college, 4: Associate degree, 5: Bachelor's degree, 6: College graduate)", options=list(range(1, 7)))
income = st.selectbox("Income level (1: <$10k, 2: $10k-$20k, 3: $20k-$30k, 4: $30k-$40k, 5: $40k-$50k, 6: $50k-$60k, 7: $60k-$75k, 8: $75k+)", options=list(range(1, 9)))
print("Feature Input Created")

# Prepare input data for prediction
input_data = np.array([[high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease, 
                        PhysActivity, fruits, veggies, heavy_alcohol, any_healthcare, no_docbc_cost,
                        gen_health, mental_health, physical_health, diff_walk, sex, age, education, income]])
print("Input Data Created")

# Scale the input data
scaled_data = scaler.transform(input_data)
print("Scaler Data Created")

# Prediction
if st.button("Predict Diabetes"):
    prediction = model.predict(scaled_data)
    result = "You Are Likely to Have Diabetes, Please Consult a Doctor" if prediction[0] == 1 else "You Are Unlikely to Have Diabetes"
    st.success(f"Prediction: {result}")
print("Prediction Done")

# Additional Information
st.write("Note: This prediction is based on a model trained on historical data and should not replace medical advice.")

print("Model is Perfectly Working")