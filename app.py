import streamlit as st
import pickle
import json
import numpy as np

import sklearn.linear_model._base
import sys

sys.modules['sklearn.linear_model.base'] = sklearn.linear_model._base
# ------------------------------
# Load Model and Columns
# ------------------------------
with open("banglore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Extract locations (skip first 3 columns)
locations = data_columns[3:]

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Real Estate Price Prediction", layout="centered")

st.title("🏠 Real Estate Price Prediction App")
st.write("Predict house prices using a trained Machine Learning model")

# ------------------------------
# Input Fields
# ------------------------------
location = st.selectbox("📍 Location", locations)

total_sqft = st.number_input(
    "📐 Total Square Feet",
    min_value=300,
    max_value=10000,
    value=1000
)

bhk = st.number_input(
    "🛏 BHK",
    min_value=1,
    max_value=10,
    value=2
)

bath = st.number_input(
    "🛁 Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# ------------------------------
# Prediction Logic
# ------------------------------
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))

    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)

# ------------------------------
# Predict Button
# ------------------------------
if st.button("🔮 Predict Price"):
    price = predict_price(location, total_sqft, bath, bhk)
    st.success(f"💰 Estimated Price: ₹ {price} Lakhs")
