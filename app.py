from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

from preprocessing.preprocessing import (
    clean_data,
    encode_data,
    prepare_for_model
)

app = Flask(__name__)

model = joblib.load("models/supply_chain_model.pkl")
encoders = joblib.load("models/encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "CSV file is required"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        df = pd.read_csv(file, encoding="latin1")

        X = clean_data(df)
        X = encode_data(X, encoders)
        X = prepare_for_model(X, model)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        results = []
        for prediction, probability in zip(predictions, probabilities):
            results.append({
                "prediction": "Late" if prediction == 1 else "On Time",
                "risk": round(float(probability) * 100, 2)
            })

        return jsonify({"results": results})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
if __name__ == "__main__":
    app.run(debug=True)