import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model once at the start
with open("water.pkl", "rb") as model_file:
    avinash = pickle.load(model_file)

#def predict(data):
    #return clf.predict(data)

st.title("Water Potability Prediction")
st.markdown("This model predicts the quality of water.")

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

if st.button("Predict Potability"):
    # Prepare input data for prediction
    input_data = np.array([[pp, Hd, So, Ch, Co, Oc, Tr, Tu, ii]])
    result = avinash.predict(input_data)

    # Display the result
    if result[0] == 1:
        st.success("The water is predicted to be potable.")
    else:
        st.error("The water is predicted not to be potable.")

st.markdown("Developed at IIIT Surat")
