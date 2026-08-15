from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Saved ML model aur encoders startup par load honge
model = joblib.load("models/supply_chain_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# Training ke time use hue exact features
FEATURES = [
    "Type",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Category Id",
    "Category Name",
    "Customer City",
    "Customer Country",
    "Customer Segment",
    "Customer State",
    "Department Id",
    "Department Name",
    "Latitude",
    "Longitude",
    "Market",
    "Order City",
    "Order Country",
    "order date (DateOrders)",
    "Order Item Cardprod Id",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Order Region",
    "Order State",
    "Product Card Id",
    "Product Category Id",
    "Product Name",
    "Product Price",
    "Shipping Mode",
    "Month",
    "Peak_Season",
    "Order_Day",
    "Is_Weekend",
    "Order_Value_Category"
]


@app.route("/")
def home():
    return render_template(
        "index.html",
        features=FEATURES,
        encoders=encoders
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    # Required fields check
    missing = [feature for feature in FEATURES if not data.get(feature)]

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "fields": missing
        }), 400

    row = {}

    try:
        for feature in FEATURES:

            value = data.get(feature)

            # Categorical column
            if feature in encoders:
                encoder = encoders[feature]

                if value not in encoder.classes_:
                    return jsonify({
                        "error": f"Unknown value '{value}' for {feature}"
                    }), 400

                row[feature] = encoder.transform([value])[0]

            # Numeric column
            else:
                row[feature] = float(value)

        # Exact training order
        import pandas as pd

        X = pd.DataFrame([row], columns=FEATURES)

        prediction = model.predict(X)[0]

        # Probability of class 1 = Late Delivery Risk
        class_index = list(model.classes_).index(1)
        risk_probability = model.predict_proba(X)[0][class_index]

        return jsonify({
            "prediction": "Late" if prediction == 1 else "On Time",
            "risk_percentage": round(risk_probability * 100, 2)
        })

    except ValueError:
        return jsonify({
            "error": "Invalid numeric value. Please enter valid numbers."
        }), 400


if __name__ == "__main__":
    app.run(debug=True)