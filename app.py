import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

st.set_page_config(page_title="Bank Term Deposit Prediction", layout="wide")

st.title("🏦 Bank Term Deposit Subscription Prediction")
st.markdown("""
Predict whether a customer will **subscribe to a term deposit**  
based on their demographic and banking information.
""")

@st.cache_resource
def load_model_and_tools():
    # Load TensorFlow or Scikit-learn model
    model = tf.keras.models.load_model("bank_model.h5")   # trained model file
    scaler = joblib.load("scaler.pkl")                    # trained scaler
    le_dict = joblib.load("label_encoders.pkl")           # label encoders
    try:
        feature_names = joblib.load("feature_names.pkl")  # optional safeguard
    except:
        feature_names = None
    return model, scaler, le_dict, feature_names

model, scaler, le_dict, feature_names = load_model_and_tools()

numeric_cols = [
    'age', 'duration', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]

categorical_cols = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'poutcome'
]


st.sidebar.header("📋 Input Customer Information")

user_input = {}

st.sidebar.subheader("Numeric Details")
for col in numeric_cols:
    user_input[col] = st.sidebar.number_input(f"{col}", value=0.0, format="%.2f")

st.sidebar.subheader("Categorical Details")
for col in categorical_cols:
    user_input[col] = st.sidebar.selectbox(f"{col}", le_dict[col].classes_)

input_df = pd.DataFrame([user_input])

for col in categorical_cols:
    input_df[col] = le_dict[col].transform(input_df[col])

if feature_names:
    input_df = input_df.reindex(columns=feature_names)

input_df_scaled = scaler.transform(input_df)


if st.button("🔮 Predict Subscription"):
    pred_prob = model.predict(input_df_scaled)
    pred_class = (pred_prob > 0.5).astype(int)[0][0]

    st.subheader("📊 Prediction Result:")
    if pred_class == 1:
        st.success(f"✅ The client is **likely to subscribe** a term deposit")
    else:
        st.error(f"❌ The client is **not likely to subscribe** a term deposit")

