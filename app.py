import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------- Title ----------------------
st.set_page_config(page_title="Breast Cancer Detection App", layout="wide")
st.title("🩺 Breast Cancer Detection using Machine Learning")

# ---------------------- Sidebar ----------------------
st.sidebar.header("Input Features")

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, data.target_names

X, y, target_names = load_data()

# ---------------------- Model Training ----------------------
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

model, acc = train_model()
st.sidebar.success(f"Model trained with accuracy: {acc*100:.2f}%")

# ---------------------- User Input ----------------------
def user_input():
    inputs = {}
    for feature in X.columns[:10]:  # First 10 features for simplicity
        inputs[feature] = st.sidebar.slider(
            feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean())
        )
    return pd.DataFrame([inputs])

input_df = user_input()

# ---------------------- Prediction ----------------------
st.subheader("🔍 Prediction Result")

prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

st.write("### Prediction:", "🧬 Malignant" if prediction == 0 else "✅ Benign")
st.write(f"Confidence: {prediction_proba[prediction]*100:.2f}%")

# ---------------------- Visualization ----------------------
st.subheader("📊 Feature Importance")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
st.bar_chart(importances)

st.info("This app uses a Random Forest model trained on the Breast Cancer Wisconsin dataset.")

# ---------------------- Footer ----------------------
st.markdown("---")
st.caption("Developed by **Nikita Pandey**, NIT Delhi | Powered by Streamlit 💻")
