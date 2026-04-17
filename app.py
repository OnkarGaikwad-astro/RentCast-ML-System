import gradio as gr
import pickle
import json
import numpy as np

model = pickle.load(open("models/best_rf_model.pkl", "rb"))
encoders = pickle.load(open("models/label_encoders.pkl", "rb"))
FEATURES = json.load(open("models/feature_order.json"))

def predict(city, location, size, bhk, bath):

    input_dict = {
        "city": encoders["city"].transform([city])[0],
        "location": encoders["location"].transform([location])[0],
        "Size_ft²": size,
        "BHK": bhk,
        "numBathrooms": bath,
        "latitude": 19.07,
        "longitude": 72.87
    }

    for f in FEATURES:
        if f not in input_dict:
            input_dict[f] = 0

    input_vec = np.array([[input_dict[f] for f in FEATURES]])
    pred = model.predict(input_vec)[0]

    return f"₹ {int(pred):,}"

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(encoders["city"].classes_.tolist(), label="City"),
        gr.Dropdown(encoders["location"].classes_.tolist(), label="Location"),
        gr.Number(label="Size (sq ft)", value=700),
        gr.Slider(1, 6, value=2, label="BHK"),
        gr.Number(label="Bathrooms", value=2)
    ],
    outputs="text",
    title="🏙️ RentCast ML System",
    description="AI-powered rent prediction"
)

interface.launch()