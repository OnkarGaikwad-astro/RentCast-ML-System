import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

st.set_page_config(page_title="UrbanNest AI", layout="centered")

@st.cache_resource
def load_artifacts():
    model = pickle.load(open("models/best_rf_model.pkl", "rb"))
    encoders = pickle.load(open("models/label_encoders.pkl", "rb"))
    features = json.load(open("models/feature_order.json"))
    return model, encoders, features

model, encoders, FEATURES = load_artifacts()

st.title("🏙️ UrbanNest AI")
st.caption("Smart Rent Prediction Platform")

# Inputs
city = st.selectbox("City", encoders["city"].classes_)
location = st.selectbox("Location", encoders["location"].classes_)
size = st.number_input("Size (sq ft)", 100, 10000, 700)
bhk = st.slider("BHK", 1, 6, 2)
bath = st.number_input("Bathrooms", 1, 10, 2)

lat = st.number_input("Latitude", value=19.07)
lon = st.number_input("Longitude", value=72.87)

st.map({"lat": [lat], "lon": [lon]})

if st.button("Predict Rent"):

    input_dict = {
        "city": encoders["city"].transform([city])[0],
        "location": encoders["location"].transform([location])[0],
        "Size_ft²": size,
        "BHK": bhk,
        "numBathrooms": bath,
        "latitude": lat,
        "longitude": lon
    }

    # Fill missing features
    for f in FEATURES:
        if f not in input_dict:
            input_dict[f] = 0

    input_vec = np.array([[input_dict[f] for f in FEATURES]])

    pred = model.predict(input_vec)[0]

    st.success(f"💰 Estimated Rent: ₹{pred:,.0f}")

    # Feature importance
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(5)

    st.subheader("🔍 Key Factors")
    st.dataframe(imp_df)