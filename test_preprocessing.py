import joblib
import pandas as pd
from preprocessing.preprocessing import clean_data, encode_data, prepare_for_model

df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin1")

model = joblib.load("models/supply_chain_model.pkl")
encoders = joblib.load("models/encoders.pkl")

X = clean_data(df)
X = encode_data(X, encoders)
X = prepare_for_model(X, model)

predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

print("Model prediction successful")
print("Rows predicted:", len(predictions))
print("Late predictions:", predictions.sum())
print("On-time predictions:", len(predictions) - predictions.sum())
print("Average risk:", round(probabilities.mean() * 100, 2), "%")
print("First 5 risks:", (probabilities[:5] * 100).round(2))