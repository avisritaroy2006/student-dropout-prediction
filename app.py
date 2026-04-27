import streamlit as st
import pandas as pd
import os

st.title("Student Dropout Predictor")

DATA_PATH = "dataset.csv"

if not os.path.exists(DATA_PATH):
    st.error("dataset.csv not found")
    st.stop()

df = pd.read_csv(DATA_PATH)

df["Target"] = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)

df["CGPA"] = (
    df["Curricular units 1st sem (grade)"] +
    df["Curricular units 2nd sem (grade)"]
) / 2

features = [
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "CGPA",
    "Tuition fees up to date",
    "Scholarship holder",
    "Age at enrollment"
]

from sklearn.ensemble import RandomForestClassifier

X = df[features]
y = df["Target"]

model = RandomForestClassifier()
model.fit(X, y)

sem1 = st.number_input("Sem1")
sem2 = st.number_input("Sem2")
grade1 = st.number_input("Grade1")
grade2 = st.number_input("Grade2")
fees = st.selectbox("Fees", [1, 0])
scholar = st.selectbox("Scholar", [1, 0])
age = st.number_input("Age")

cgpa = (grade1 + grade2) / 2

if st.button("Predict"):
    data = pd.DataFrame([[sem1, sem2, grade1, grade2, cgpa, fees, scholar, age]], columns=features)
    pred = model.predict(data)

    if pred[0] == 1:
        st.error("Dropout Risk")
    else:
        st.success("No Risk")
