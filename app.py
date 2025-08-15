import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Breast Cancer Classifier",
                   page_icon="🩺",
                   layout="centered",
                   initial_sidebar_state="expanded")

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

models = {
    "Without Feature Selection": {
        "Logistic Regression": load_model("logistic_regression_all.pkl"),
        "Random Forest": load_model("random_forest_All.pkl")
    },
    "With Feature Selection": {
        "Logistic Regression": load_model("logistic_regression_selected.pkl"),
        "Random Forest": load_model("random_forest_selected.pkl")
    }
}

# ---------------------------
# UI Styling
# ---------------------------
st.markdown("""
    <style>
        .main { background-color: #F8FAFC; }
        h1, h2, h3 { text-align: center; color: #0f172a; }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-size: 1em;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Breast Cancer Prediction App")
st.write("Enter patient features below and select the model to make predictions.")

# ---------------------------
# Sidebar - Model Selection
# ---------------------------
st.sidebar.header("⚙️ Model Settings")
feature_mode = st.sidebar.radio("Feature Selection", ["Without Feature Selection", "With Feature Selection"])
model_choice = st.sidebar.radio("Choose Model", ["Logistic Regression", "Random Forest"])

# ---------------------------
# Feature Input Form
# ---------------------------
st.subheader("📊 Input Patient Data")

# NOTE: Replace these with your actual dataset feature names
all_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

user_input = {}
cols = st.columns(3)
for i, feature in enumerate(all_features):
    with cols[i % 3]:
        user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")

# Convert input to array
input_df = pd.DataFrame([user_input])

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict"):
    model = models[feature_mode][model_choice]
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader("📢 Prediction Result")
    if prediction == 1:
        st.success("✅ The model predicts **Malignant**")
    else:
        st.info("🔵 The model predicts **Benign**")

    st.subheader("📈 Prediction Probabilities")
    st.write(f"Benign: {probabilities[0]*100:.2f}%")
    st.write(f"Malignant: {probabilities[1]*100:.2f}%")
