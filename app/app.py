# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('house_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler object used for normalization

# Function to predict house price
def predict_price(features):
    features_scaled = scaler.transform([features])
    predicted_price = model.predict(features_scaled)[0]
    return predicted_price

# Main function to run the app
def main():
    st.title("Boston Housing Price Prediction")
    st.write("This app predicts the median house price in Boston based on various features.")
    
    # Add background image using custom CSS
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url('https://www.vantage-ai.com/hubfs/House%20prices.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar with input fields
    st.sidebar.header("Input Features")
    
    crim = st.sidebar.number_input("CRIM (per capita crime rate)", value=0.0)
    zn = st.sidebar.number_input("ZN (proportion of residential land zoned for lots over 25,000 sq.ft.)", value=0.0)
    indus = st.sidebar.number_input("INDUS (proportion of non-retail business acres per town)", value=0.0)
    chas = st.sidebar.selectbox("CHAS (Charles River dummy variable)", [0, 1])
    nox = st.sidebar.number_input("NOX (nitric oxides concentration)", value=0.0)
    rm = st.sidebar.number_input("RM (average number of rooms per dwelling)", value=0.0)
    age = st.sidebar.number_input("AGE (proportion of owner-occupied units built prior to 1940)", value=0.0)
    dis = st.sidebar.number_input("DIS (weighted distances to five Boston employment centres)", value=0.0)
    rad = st.sidebar.number_input("RAD (index of accessibility to radial highways)", value=0.0)
    tax = st.sidebar.number_input("TAX (full-value property-tax rate per $10,000)", value=0.0)
    ptratio = st.sidebar.number_input("PTRATIO (pupil-teacher ratio by town)", value=0.0)
    b = st.sidebar.number_input("B (proportion of African Americans by town)", value=0.0)
    lstat = st.sidebar.number_input("LSTAT (% lower status of the population)", value=0.0)
    
    # Predict house price when the user clicks the button
    if st.sidebar.button("Predict"):
        features = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]
        predicted_price = predict_price(features)
        st.success(f"Predicted Median House Price: ${predicted_price:.2f}")

if __name__ == "__main__":
    main()
