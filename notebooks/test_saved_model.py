import pandas as pd
import joblib

model = joblib.load("models/supply_chain_model.pkl")
encoders = joblib.load("models/encoders.pkl")

print("Model loaded successfully")
print("Encoders loaded successfully")

df = pd.read_csv("notebooks/cleaned_supply_chain.csv")

X = df.drop(columns=[
    "Late_delivery_risk",
    "Delay_Days",
    "Delay_Category",
    "Days for shipping (real)",
    "Delivery Status"
])

for col, encoder in encoders.items():
    X[col] = encoder.transform(X[col])

X = X[model.feature_names_in_]

prediction = model.predict(X.iloc[[0]])

print("Prediction:", prediction[0])
print(model.feature_names_in_.tolist())