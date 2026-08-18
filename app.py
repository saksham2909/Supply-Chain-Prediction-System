from time import time

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import time

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

        # X = clean_data(df)
        # X = encode_data(X, encoders)
        # X = prepare_for_model(X, model)
        start = time.time()

        X = clean_data(df)
        print("clean_data:", time.time() - start)

        start = time.time()
        X = encode_data(X, encoders)
        print("encode_data:", time.time() - start)

        start = time.time()
        X = prepare_for_model(X, model)
        print("prepare_for_model:", time.time() - start)
        
        start = time.time()
        predictions = model.predict(X)
        print("Rows:", len(X))
        print("Features:", X.shape[1])
        probabilities = model.predict_proba(X)[:, 1]
        print("Prediction time:", time.time() - start)
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