import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier
)
from sklearn.neighbors import KNeighborsClassifier

# Load and prepare data
HDNames = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'HeartDisease']
data = pd.read_excel('Ch3.ClevelandData.xlsx', names=HDNames)
data_new = data.replace("?", np.nan).dropna()

feature_names = HDNames[:-1]
features = data_new[feature_names]
target = data_new['HeartDisease']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.30, random_state=5)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
}

for model in models.values():
    model.fit(X_train, y_train)

# Function to predict heart disease
def predict_heart_disease(input_data, model):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return prediction[0]

# Create UI
st.title("Heart Disease Prediction")
st.write("### Enter your health information:")
st.write("Please fill in the details below:")

# Input fields for user data
age = st.number_input("Age (1-100)", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex (1 = male; 0 = female)", [0, 1])
cp = st.selectbox("Chest Pain Type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic)", [1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=0)
exang = st.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, format="%.1f")
slope = st.selectbox("Slope of the Peak Exercise ST Segment (1 = upsloping, 2 = flat, 3 = downsloping)", [1, 2, 3])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-4)", min_value=0, max_value=4)
thal = st.selectbox("Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)", [3, 6, 7])

input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Button to submit the input data
if st.button("Submit"):
    # Validate inputs
    valid = True
    error_messages = []

    if age < 1 or age > 100:
        valid = False
        error_messages.append("Age must be between 1 and 100.")
    if cp not in [1, 2, 3, 4]:
        valid = False
        error_messages.append("Chest Pain Type must be 1, 2, 3, or 4.")
    if restecg not in [0, 1, 2]:
        valid = False
        error_messages.append("Resting Electrocardiographic Results must be 0, 1, or 2.")
    if slope not in [1, 2, 3]:
        valid = False
        error_messages.append("Slope must be 1, 2, or 3.")
    if thal not in [3, 6, 7]:
        valid = False
        error_messages.append("Thalassemia must be 3, 6, or 7.")

    if valid:
        predictions = {}
        
        # Store predictions from all models
        for model_name, model in models.items():
            prediction = predict_heart_disease(input_data, model)
            predictions[model_name] = prediction
        
        # Display predictions and count "Yes" results
        yes_count = sum(pred == 1 for pred in predictions.values())
        no_count = len(predictions) - yes_count
        
        st.write("### Model Predictions:")
        for model_name, prediction in predictions.items():
            if prediction == 1:
                st.success(f"**{model_name}:** High chance of heart disease")
            else:
                st.warning(f"**{model_name}:** Low chance of heart disease")
        
        st.write("### Summary:")
        st.write(f"Total 'Yes' predictions: **{yes_count}**")
        st.write(f"Total 'No' predictions: **{no_count}**")
        
        if yes_count > 0:
            st.write("### Health Recommendations:")
            st.success("**You should consider the following tips for heart health:**")
            st.write("- Maintain a balanced diet rich in fruits, vegetables, and whole grains.")
            st.write("- Engage in regular physical activity.")
            st.write("- Avoid smoking and limit alcohol consumption.")
            st.write("- Manage stress through mindfulness and relaxation techniques.")
            st.write("- Regular check-ups with your healthcare provider.")
            
            st.write("**Get tested:** [Heart Care India](https://www.heartcareindia.com)")
        else:
            st.write("### Health Recommendations:")
            st.info("**You are advised to maintain a healthy lifestyle:**")
            st.write("- Continue to maintain a healthy lifestyle.")
            st.write("- Monitor your blood pressure and cholesterol levels.")
            st.write("- Stay physically active and eat a balanced diet.")
    else:
        st.error("Please correct the following errors:")
        for message in error_messages:
            st.write(f"- {message}")
