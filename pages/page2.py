import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

model = joblib.load("models/XGBoost_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

st.title("🎯 Obesity Prediction Demo (XGBoost Model)")
st.write("Fill out the user's information to the executive level of obesity.")

form = st.form("prediction_form")
Gender = form.selectbox("Gender", label_encoders["Gender"].classes_)
Age = form.slider("Age", 10, 100, 25)
Height = form.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.70)
Weight = form.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=60.0)
family_history_with_overweight = form.selectbox("Family History with Overweight", label_encoders["family_history_with_overweight"].classes_)
FAVC = form.selectbox("Frequent consumption of high caloric food (FAVC)", label_encoders["FAVC"].classes_)
FCVC = int(form.slider("Frequency of vegetable consumption (FCVC)", 1, 3, 2))
NCP = int(form.slider("Number of main meals (NCP)", 1, 4, 3))
CAEC = form.selectbox("Consumption of food between meals (CAEC)", label_encoders["CAEC"].classes_)
SMOKE = form.selectbox("Do you smoke?", label_encoders["SMOKE"].classes_)
CH2O = int(form.slider("Daily water consumption (CH2O)", 1, 3, 2))
SCC = form.selectbox("Calories monitoring (SCC)", label_encoders["SCC"].classes_)
FAF = form.slider("Physical activity frequency (FAF)", 0, 3, 1)
#TUE = int(form.slider("Time using technology devices (TUE)", 0, 2, 1))
CALC = form.selectbox("Alcohol consumption (CALC)", label_encoders["CALC"].classes_)
MTRANS = form.selectbox("Transportation used", label_encoders["MTRANS"].classes_)
submit = form.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([{
        "Gender": Gender,
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "family_history_with_overweight": family_history_with_overweight,
        "FAVC": FAVC,
        "FCVC": FCVC,
        "NCP": NCP,
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "CH2O": CH2O,
        "SCC": SCC,
        "FAF": FAF,
        #"TUE": TUE,
        "CALC": CALC,
        "MTRANS": MTRANS
    }])

    for col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_label = target_encoder.inverse_transform(prediction)

    st.success(f"🩺 Prediction results: **{prediction_label[0]}**")

