import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel

# Load the dataset
@st.cache
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    data = pd.read_csv(url, names=columns)
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)
    data = data.astype(float)
    return data

# Load data
data = load_data()

# Preprocess the data
X = data.drop('target', axis=1)
y = data['target']
y = y.apply(lambda x: 1 if x > 0 else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a new model for feature importance
clf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
clf_importance.fit(X_train, y_train)

# Get feature importances
importances = clf_importance.feature_importances_
feature_names = X.columns


# User input
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.sidebar.selectbox('Sex', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    trestbps = st.sidebar.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
    chol = st.sidebar.number_input('Serum Cholestoral in mg/dl (chol)', min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', [0, 1, 2])
    thalach = st.sidebar.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.sidebar.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment (slope)', [0, 1, 2])
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flourosopy (ca)', [0, 1, 2, 3])
    thal = st.sidebar.selectbox('Thal', [1, 2, 3])
    
    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess user input
input_scaled = scaler.transform(input_df)

# Predict using the trained model
prediction = clf_importance.predict(input_scaled)
prediction_proba = clf_importance.predict_proba(input_scaled)

# Display user input
st.subheader('User Input parameters')
st.write(input_df)

# Display prediction
st.subheader('Prediction')
heart_disease = np.array(['No Heart Disease', 'Heart Disease'])
st.write(heart_disease[prediction])

# Display prediction probability
st.subheader('Prediction Probability')
st.write(prediction_proba)
