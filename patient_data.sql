
CREATE DATABASE heart_disease_db;
USE heart_disease_db;
CREATE TABLE patient_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age INT,
    sex INT,
    cp INT,
    trestbps INT,
    chol INT,
    fbs INT,
    restecg INT,
    thalach INT,
    exang INT,
    oldpeak FLOAT,
    slope INT,
    ca INT,
    thal INT,
    prediction VARCHAR(50),
    probability FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
SELECT * FROM heart_disease_db.patient_data;