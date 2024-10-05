# Heart-Disease-predicting-app

## Purpose

This application predicts the likelihood of heart disease based on various medical parameters. It leverages machine learning to provide early detection and help in preventive healthcare.
## Live Demo

You can check out the live version of the project [here](https://heartpredict.streamlit.app/).



## Tech Stack

- **Streamlit**: For creating a user-friendly web interface.
- **scikit-learn**: For building and evaluating the machine learning model.
- **pandas**: For data manipulation.
- **numpy**: For numerical operations.

## Key Features

- Interactive web interface for entering medical parameters.
- Uses a RandomForestClassifier for predictions.
- Provides accuracy and a detailed classification report for the model.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Rivu49/HeartDiseasePrediction.git
    cd Heart-Disease-predicting-app-main
    ```

2. **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Model Details

The heart disease prediction model is a `RandomForestClassifier`. The model parameters are tuned for optimal performance. Below are the key parameters used:

- `n_estimators=1000`: Number of trees in the forest.
- `max_depth=15`: Maximum depth of the tree.
- `min_samples_split=5`: Minimum number of samples required to split an internal node.
- `min_samples_leaf=2`: Minimum number of samples required to be at a leaf node.
- `max_features='sqrt'`: Number of features to consider when looking for the best split.
- `class_weight='balanced'`: Adjusts weights inversely proportional to class frequencies.
- `random_state=42`: Seed for the random number generator.

## Usage

Once the application is running, you can access it via your web browser. Enter the required medical parameters to get a prediction on the likelihood of heart disease.

### Input Parameters

- **Age**: Age of the patient.
- **Sex**: Gender of the patient (Male/Female).
- **Chest Pain Type**: Type of chest pain experienced.
- **Resting Blood Pressure**: Resting blood pressure in mm Hg.
- **Serum Cholesterol**: Serum cholesterol in mg/dl.
- **Fasting Blood Sugar**: Fasting blood sugar > 120 mg/dl (True/False).
- **Resting ECG Results**: Results of resting electrocardiographic test.
- **Max Heart Rate Achieved**: Maximum heart rate achieved during exercise.
- **Exercise Induced Angina**: Exercise-induced angina (Yes/No).
- **ST Depression**: ST depression induced by exercise relative to rest.
- **Slope of Peak Exercise ST Segment**: Slope of the peak exercise ST segment.
- **Number of Major Vessels**: Number of major vessels colored by fluoroscopy.
- **Thal**: Thalassemia (Normal, Fixed Defect, Reversible Defect).

### Example Code

Here’s a snippet of the code used in `app.py`:

```python
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Title
st.title("Heart Disease Prediction App")

# Input fields
age = st.number_input("Age", 1, 100)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.number_input("Chest Pain Type", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 0, 200)
chol = st.number_input("Serum Cholestoral in mg/dl", 0, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.number_input("Resting Electrocardiographic Results", 0, 2)
thalach = st.number_input("Maximum Heart Rate Achieved", 0, 220)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 10.0)
slope = st.number_input("Slope of the peak exercise ST segment", 0, 2)
ca = st.number_input("Number of major vessels colored by flourosopy", 0, 3)
thal = st.number_input("thal", 0, 3)

# Convert categorical variables to numeric
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "True" else 0
exang = 1 if exang == "Yes" else 0

# Prediction
if st.button("Predict"):
    features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("The model predicts that you have heart disease.")
    else:
        st.success("The model predicts that you do not have heart disease.")

