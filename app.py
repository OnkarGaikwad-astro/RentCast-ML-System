import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# ── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="UrbanNest AI", layout="wide")

# ── Custom CSS (THE MAGIC ✨) ─────────────────────────────
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}
.big-font {
    font-size: 38px !important;
    font-weight: bold;
    color: #4ade80;
}
.subtle {
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("models/best_rf_model.pkl", "rb"))
    encoders = pickle.load(open("models/label_encoders.pkl", "rb"))
    features = json.load(open("models/feature_order.json"))
    return model, encoders, features

model, encoders, FEATURES = load_artifacts()

# ── Header ───────────────────────────────────────────────
st.markdown("<h1>🏙️ UrbanNest AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtle'>Smart Rent Intelligence Platform</p>", unsafe_allow_html=True)

st.divider()

# ── Layout ───────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

# ── LEFT PANEL ───────────────────────────────────────────
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📍 Location Details")

    city = st.selectbox("City", encoders["city"].classes_)
    location = st.selectbox("Area", encoders["location"].classes_)

    st.markdown("### 🌍 Coordinates")
    lat = st.number_input("Latitude", value=19.07)
    lon = st.number_input("Longitude", value=72.87)

    st.map({"lat": [lat], "lon": [lon]})

    st.markdown("</div>", unsafe_allow_html=True)

# ── RIGHT PANEL ──────────────────────────────────────────
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏠 Property Details")

    size = st.number_input("Size (sq ft)", 100, 10000, 700)
    bhk = st.slider("BHK", 1, 6, 2)
    bath = st.number_input("Bathrooms", 1, 10, 2)
    balconies = st.number_input("Balconies", 0, 5, 1)
    rooms = st.number_input("Total Rooms", 1, 15, 3)

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ── Predict Button ───────────────────────────────────────
predict = st.button("🔮 Predict Rent", use_container_width=True)

# ── Prediction Section ───────────────────────────────────
if predict:

    input_dict = {
        "city": encoders["city"].transform([city])[0],
        "location": encoders["location"].transform([location])[0],
        "Size_ft²": size,
        "BHK": bhk,
        "numBathrooms": bath,
        "numBalconies": balconies,
        "rooms_num": rooms,
        "latitude": lat,
        "longitude": lon
    }

    for f in FEATURES:
        if f not in input_dict:
            input_dict[f] = 0

    input_vec = np.array([[input_dict[f] for f in FEATURES]])
    pred = model.predict(input_vec)[0]

    # ── RESULT CARD ───────────────────────────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 💰 Estimated Monthly Rent")
    st.markdown(f"<p class='big-font'>₹ {pred:,.0f}</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtle'>Based on your inputs</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── FEATURE IMPORTANCE ───────────────────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Key Influencing Factors")

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(5)

    st.bar_chart(imp_df.set_index("Feature"))

    st.markdown("</div>", unsafe_allow_html=True)

    # ── INPUT DATA ───────────────────────────────────────
    with st.expander("📊 View Input Details"):
        st.dataframe(pd.DataFrame(input_vec, columns=FEATURES))