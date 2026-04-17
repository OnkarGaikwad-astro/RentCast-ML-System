import pandas as pd
import numpy as np
import pickle
import json
import optuna
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load data
train_df = pd.read_csv("Dataset/train.csv")
test_df  = pd.read_csv("Dataset/test.csv")
print(train_df.columns)

# Separate target
TARGET = "price"
X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]
print("Target column:", TARGET)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
# Encode categorical columns
encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Save feature order
FEATURES = list(X.columns)

# ---------------- GRID SEARCH ----------------
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [10, 15, 20, 25, 30],
    "min_samples_split": [2, 5, 8]
}

grid = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid.fit(X, y)
print("Grid Best:", grid.best_params_)

# ---------------- RANDOM SEARCH ----------------
param_dist = {
    "n_estimators": np.arange(50, 201),
    "max_depth": np.arange(10, 31),
    "min_samples_split": np.arange(2, 11)
}

random = RandomizedSearchCV(
    RandomForestRegressor(),
    param_dist,
    n_iter=60,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    random_state=42
)

random.fit(X, y)
print("Random Best:", random.best_params_)

# ---------------- OPTUNA ----------------
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 10, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
    }

    model = RandomForestRegressor(**params)

    score = cross_val_score(
        model, X, y,
        cv=5,
        scoring="neg_mean_absolute_error"
    ).mean()

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60)

print("Optuna Best:", study.best_params)

# ---------------- FINAL MODEL ----------------
best_params = study.best_params

final_model = RandomForestRegressor(**best_params)
final_model.fit(X, y)

# Evaluate on test set
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# Encode test using same encoders
def safe_transform(le, values):
    known = set(le.classes_)
    return [le.transform([v])[0] if v in known else -1 for v in values]

for col in encoders:
    X_test[col] = safe_transform(encoders[col], X_test[col])

preds = final_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("Final MAE:", mae)

# ---------------- SAVE EVERYTHING ----------------
pickle.dump(final_model, open("models/best_rf_model.pkl", "wb"))
pickle.dump(encoders, open("models/label_encoders.pkl", "wb"))

with open("models/feature_order.json", "w") as f:
    json.dump(FEATURES, f)

print("✅ Model saved successfully!")