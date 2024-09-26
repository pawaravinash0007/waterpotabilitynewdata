import streamlit as st
import numpy as np
import joblib

# Load the model once at the start
model = joblib.load(open("water111.pkl", "rb"))

st.title("Water Potability Prediction")
st.markdown("This model predicts the quality of water based on various parameters.")

st.header("Input Parameters")
col1, col2 = st.columns(2)

with col1:
    pp = st.slider("pH Value", 1.0, 14.0, 7.0)  # Default to neutral pH
    Hd = st.slider("Hardness Value", 65.0, 330.0, 150.0)  # Default value
    So = st.slider("Solids Value", 600.0, 62000.0, 1000.0)  # Default value
    Ch = st.slider("Chloramines Value", 0.0, 14.0, 2.0)  # Default value
    Su = st.slider("Sulfate Value", 100.0, 500.0, 200.0)  # Default value

with col2:
    Co = st.slider("Conductivity Value", 150.0, 800.0, 300.0)  # Default value
    Oc = st.slider("Organic Carbon Value", 2.0, 30.0, 5.0)  # Default value
    Tr = st.slider("Trihalomethanes Value", 5.0, 130.0, 10.0)  # Default value
    Tu = st.slider("Turbidity Value", 1.0, 7.0, 2.0)  # Default value
    ii = st.slider("ID", 1.0, 3280.0, 1.0)  # Default value

# Prepare input data for prediction
input_data = np.array([[pp, Hd, So, Ch, Su, Co, Oc, Tr, Tu, ii]])

# Prediction button
if st.button("Predict Potability"):
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction result
    if prediction[0] == 1:
        st.write("Prediction: The water is **potable**.")
    else:
        st.write("Prediction: The water is **not potable**.")
