import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Título del aplicativo
st.title("Predicción de Diabetes con Machine Learning")

# 2. Cargar datos
url = "https://raw.githubusercontent.com/LuisPerezTimana/Webinars/main/diabetes.csv"
df = pd.read_csv(url)

st.subheader("Vista previa de los datos")
st.write(df.head())

# 3. Selección de variables
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Escalado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Entrenar modelo
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 7. Evaluación
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Evaluación del modelo")
st.write(f"Exactitud: {acc:.2f}")
st.write("Matriz de confusión:")
st.write(confusion_matrix(y_test, y_pred))

# 8. Predicción interactiva
st.subheader("Haz tu propia predicción")

# Crear entradas para cada variable
pregnancies = st.number_input("Número de embarazos", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucosa", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Presión sanguínea", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Espesor de piel", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulina", min_value=0, max_value=900, value=80)
bmi = st.number_input("Índice de masa corporal (BMI)", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Edad", min_value=0, max_value=120, value=30)

# Botón de predicción
if st.button("Predecir"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)
    prediction = rf.predict(input_data)[0]

    if prediction == 1:
        st.error("El modelo predice: **Diabético**")
    else:
        st.success("El modelo predice: **No diabético**")
