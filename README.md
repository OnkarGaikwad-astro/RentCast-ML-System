# 🏙️ RentCast ML System — Smart Rent Prediction Platform

🚀 **Live Demo:** https://huggingface.co/spaces/Astronkar/rentcast-ml-system

---

## 📌 Overview

RentCast ML System is an end-to-end machine learning application designed to predict residential rental prices based on property features, location, and amenities.

This project simulates a real-world PropTech solution by combining model optimization, deployment, and an interactive user interface.

---

## ✨ Features

* 🔮 Predict rental prices using machine learning
* ⚙️ Hyperparameter tuning using:

  * Grid Search
  * Random Search
  * Bayesian Optimization (Optuna)
* 📊 Feature importance visualization
* 🌍 Interactive map-based location input
* 🧠 Handles unseen categorical values
* 🖥️ Clean and responsive Streamlit UI
* 🐳 Dockerized deployment for reproducibility

---

## 🧠 Tech Stack

* **Language:** Python
* **Libraries:**

  * Pandas, NumPy
  * Scikit-learn
  * Optuna
  * Streamlit
* **Deployment:**

  * Hugging Face Spaces (Docker)

---

## 📂 Project Structure

```
RentCast-ML-System/
├── app.py                  # Streamlit UI
├── train.py               # Model training & optimization
├── Dockerfile             # Deployment config
├── requirements.txt       # Dependencies
├── models/                # Saved model & encoders
│   ├── best_rf_model.pkl
│   ├── label_encoders.pkl
│   └── feature_order.json
├── Dataset/               # Training & testing data
└── README.md
```

---

## 📊 Model Details

* **Algorithm:** Random Forest Regressor
* **Optimization:** Bayesian Optimization (Optuna)
* **Evaluation Metric:** Mean Absolute Error (MAE)

### 📈 Final Performance

* **MAE:** ₹12,438
* **Best Parameters:**

  * `n_estimators`: 170
  * `max_depth`: 28
  * `min_samples_split`: 2

---

## 🚀 How to Run Locally

```bash
# Clone repository
git clone https://github.com/your-username/RentCast-ML-System.git
cd RentCast-ML-System

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 🐳 Run with Docker

```bash
docker build -t rentcast .
docker run -p 7860:7860 rentcast
```

---

## 🌐 Deployment

The application is deployed using Hugging Face Spaces with Docker, enabling seamless cloud access and real-time predictions.

---

## 🧾 Key Learnings

* End-to-end ML pipeline development
* Hyperparameter optimization under compute constraints
* Handling unseen categorical data
* Model deployment using Streamlit and Docker
* Building user-friendly ML interfaces

---

## 👨‍💻 Author

**Onkar Gaikwad**
🔗 GitHub: https://github.com/OnkarGaikwad-astro

---

## ⭐ If you found this useful

Give this repo a star ⭐ and share it!
