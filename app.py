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
model = RandomForestClassifier(min_samples_split=100,min_samples_leaf=4,min_depth=3,n_estimators=100)
model.fit(X_train, y_train)

# Step 5: Save the model
joblib.dump(model, "mymodel_iiit.pkl")

# Step 6: Start the Streamlit app
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
