import numpy as np
import pandas as pd
import streamlit as st
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Connect to SQLite database
conn = sqlite3.connect("heart_disease.db", check_same_thread=False)
cursor = conn.cursor()

# Create table for storing patient data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        sex INTEGER,
        cp INTEGER,
        trestbps INTEGER,
        chol INTEGER,
        fbs INTEGER,
        restecg INTEGER,
        thalach INTEGER,
        exang INTEGER,
        oldpeak REAL,
        slope INTEGER,
        ca INTEGER,
        thal INTEGER,
        prediction INTEGER
    )
''')
conn.commit()

# Load dataset
url = 'D:\\heartdise\\healthdata.csv'  # Change if needed
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=column_names)

data.replace('?', np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(data.mean(), inplace=True)
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

X = data.drop('target', axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Streamlit UI
st.title(" 🫀 Heart Disease Prediction")

# Sidebar input form
st.sidebar.header("Patient Health Details")
age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex (0-Female, 1-Male)", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 50, 250, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", [0, 1])
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])

# Prepare data for prediction
new_data = pd.DataFrame({
    'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
    'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
    'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
})

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
prediction_prob = model.predict_proba(new_data_scaled)[:, 1]

# Store patient data
def save_patient_data(data):
    cursor.execute('''
        INSERT INTO patients (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()

# Predict & Save
if st.sidebar.button("Predict & Save"):
    pred_value = int(prediction[0])
    save_patient_data((age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, pred_value))

    if pred_value == 1:
        st.error("⚠️ The patient **has heart disease**.")
    else:
        st.success("✅ The patient **does NOT have heart disease**.")
    
    st.write("**Prediction Confidence:**", round(prediction_prob[0] * 100, 2), "%")

# Show stored data with readable prediction column
if st.sidebar.button("Show Stored Data"):
    cursor.execute("SELECT * FROM patients")
    records = cursor.fetchall()

    if records:
        df = pd.DataFrame(records, columns=['ID', 'Age', 'Sex', 'CP', 'BP', 'Chol', 'FBS', 'ECG', 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'CA', 'Thal', 'Prediction'])
        
        # Replace prediction 1/0 with readable text
        df['Prediction'] = df['Prediction'].apply(lambda x: "Heart Disease" if x == 1 else "No Heart Disease")
        
        st.write(df)
    else:
        st.write("No patient records found.")

# Close database connection
conn.close()
