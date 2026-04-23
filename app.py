import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Diabetes Prediction FL", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Powered by Federated Learning Concept")

# Input fields
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# Train model (simple demo)
@st.cache_resource
def train_model():
    df = pd.read_csv("data/diabetes.csv")
    
    if "Outcome" in df.columns:
        target = "Outcome"
    else:
        target = df.columns[-1]

    X = df.drop(target, axis=1)
    y = df[target]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return model

model = train_model()

# Prediction
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Non-Diabetic")