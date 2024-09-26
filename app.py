import streamlit as st
import pandas as pd
import numpy as np
import pickle

clf = pickle.load(open("water.pkl","rb"))

def predict(data):    
    return clf.predict(data)


st.title("Water Potability Prediction")
st.markdown("This Model Predict the Quality of water")

st.header("DFADS")
col1,col2 = st.columns(2)

with col1:
	
	p1 = st.slider("Ph Value", 1.0, 14.0, 0.5)
	Hd = st.slider("Hardness Value", 65.0, 330.0, 0.5)
	So = st.slider("Solids Value", 600.0,62000.0,0.5)
	Ch = st.slider("Chloramines Value", 0.0, 14.0, 0.5)
	Su = st.slider("Sulfate Value", 100.0, 500.0, 0.5)
with col2:
	Co = st.slider("Conductivity Value", 150.0,800.0,0.5)
	Oc = st.slider("Organic Carbon Value", 2.0, 30.0, 0.5)
	Tr = st.slider("Trihalomethanes Value", 5.0, 130.0, 0.5)
	Tu = st.slider("Turbidity Value", 1.0, 7.0, 0.5)	
	i1 = st.slider("id", 1.0, 3280.0, 0.5)
st.text('')
if st.button("Seles Prediction "):
	result = clf.predict(     
	    np.array([[p1,Hd,So,Ch,Co,Oc,Tr,Tu,i1]]))
    st.text(result[0])

st.markdown("Work at IIIT Surat")
