import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st
import numpy as np

# Step 1: Load the dataset
# Replace this with the path to your actual dataset
data = pd.read_csv("ready.csv")

# Step 2: Prepare the data
X = data.drop("Potability", axis=1)  # Assuming 'Potability' is the target column
y = data["Potability"]

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest model
model = RandomForestClassifier(min_samples_split=100,min_samples_leaf=4,max_depth=5,n_estimators=100)
model.fit(X_train, y_train)

# Step 5: Save the model
joblib.dump(model, "mymodel_iiit.pkl")

# Step 6: Start the Streamlit app
st.title("Water Potability Prediction")
st.markdown("This model predicts the quality of water based on various parameters.")

st.header("Input Parameters")
col1, col2 = st.columns(2)

with col1:
    pp = st.text_input("pH Value", value=7.0)  # Text input for pH
    Hd = st.selectbox("Hardness Value", [65, 100, 150, 200, 250, 300])  # Select box for Hardness
    So = st.number_input("Solids Value", min_value=600.0, max_value=62000.0, value=1000.0)  # Number input for Solids
    Ch = st.radio("Chloramines Value", options=[0.0, 2.0, 5.0, 10.0])  # Radio button for Chloramines
    Su = st.slider("Sulfate Value", 100.0, 500.0, 200.0)  # Slider for Sulfate

with col2:
    Co = st.text_input("Conductivity Value", value=300.0)  # Text input for Conductivity
    Oc = st.selectbox("Organic Carbon Value", [2, 5, 10, 15, 20, 25, 30])  # Select box for Organic Carbon
    Tr = st.number_input("Trihalomethanes Value", min_value=5.0, max_value=130.0, value=10.0)  # Number input for Trihalomethanes
    Tu = st.radio("Turbidity Value", options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # Radio button for Turbidity
    ii = st.slider("ID", 1.0, 3280.0, 1.0)  # Slider for ID

# Prepare input data for prediction
input_data = np.array([[pp, Hd, So, Ch, Su, Co, Oc, Tr, Tu, ii,1]])

# Load the model for predictions
model = joblib.load("mymodel_iiit.pkl")

# Prediction button
if st.button("Predict Potability"):
    try:
        # Make prediction using the model
        prediction = model.predict(input_data)
        st.write(prediction[0])
        # Display prediction result
        if prediction[0] == 1:
            st.write("Prediction: The water is **potable**.")
        else:
            st.write("Prediction: The water is **not potable**.")
    except Exception as e:
        st.error(f"Error: {e}")
