"""
Supply Chain Delay Predictor - Flask Web Application
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Supply Chain Delay Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; min-height: 100vh; }
        .header {
            background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
            color: white; padding: 30px 20px; text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { font-size: 14px; opacity: 0.85; }
        .container { max-width: 750px; margin: 40px auto; padding: 0 20px; }
        .card {
            background: white; border-radius: 12px;
            padding: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .card h2 { color: #1a237e; margin-bottom: 24px; font-size: 20px; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        .form-group { display: flex; flex-direction: column; gap: 6px; }
        .form-group label { font-size: 13px; font-weight: 600; color: #555; }
        .form-group select, .form-group input {
            padding: 10px 14px; border: 1.5px solid #ddd;
            border-radius: 8px; font-size: 14px; outline: none;
            transition: border-color 0.2s;
        }
        .form-group select:focus, .form-group input:focus { border-color: #1a237e; }
        .btn {
            margin-top: 24px; width: 100%; padding: 14px;
            background: linear-gradient(135deg, #1a237e, #3949ab);
            color: white; border: none; border-radius: 8px;
            font-size: 16px; font-weight: 600; cursor: pointer;
            transition: transform 0.1s, box-shadow 0.2s;
        }
        .btn:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(26,35,126,0.3); }
        .result { margin-top: 28px; border-radius: 12px; padding: 24px; text-align: center; }
        .result.delayed { background: #fff5f5; border: 2px solid #fc8181; }
        .result.ontime  { background: #f0fff4; border: 2px solid #68d391; }
        .result-icon { font-size: 48px; margin-bottom: 10px; }
        .result-title { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
        .result.delayed .result-title { color: #c53030; }
        .result.ontime  .result-title { color: #276749; }
        .result-subtitle { font-size: 14px; color: #666; }
        .kpi-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 30px; }
        .kpi { background: #1a237e; color: white; border-radius: 10px; padding: 16px; text-align: center; }
        .kpi-val { font-size: 22px; font-weight: 700; }
        .kpi-label { font-size: 12px; opacity: 0.8; margin-top: 4px; }
        @media (max-width: 500px) { .form-grid { grid-template-columns: 1fr; } .kpi-row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚚 Supply Chain Delay Predictor</h1>
        <p>Enter order details to predict if delivery will be On Time or Delayed</p>
    </div>
    <div class="container">
        <div class="kpi-row">
            <div class="kpi"><div class="kpi-val">2,000</div><div class="kpi-label">Total Orders</div></div>
            <div class="kpi"><div class="kpi-val">34.9%</div><div class="kpi-label">Delayed Orders</div></div>
            <div class="kpi"><div class="kpi-val">7.3 Days</div><div class="kpi-label">Avg Delivery Time</div></div>
        </div>
        <div class="card">
            <h2>🔍 Predict Delivery Status</h2>
            <div class="form-grid">
                <div class="form-group">
                    <label>Shipping Mode</label>
                    <select id="shipping_mode">
                        <option value="Express">Express</option>
                        <option value="Standard">Standard</option>
                        <option value="Same Day">Same Day</option>
                        <option value="Economy">Economy</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Order Priority</label>
                    <select id="order_priority">
                        <option value="High">High</option>
                        <option value="Critical">Critical</option>
                        <option value="Medium">Medium</option>
                        <option value="Low">Low</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Distance (km)</label>
                    <input type="number" id="distance" value="500" min="50" max="2500">
                </div>
                <div class="form-group">
                    <label>Order Quantity</label>
                    <input type="number" id="quantity" value="50" min="1" max="200">
                </div>
                <div class="form-group">
                    <label>Warehouse Location</label>
                    <select id="warehouse">
                        <option>Mumbai</option><option>Delhi</option>
                        <option>Chennai</option><option>Kolkata</option>
                        <option>Hyderabad</option><option>Pune</option>
                        <option>Ahmedabad</option><option>Bangalore</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Order Month</label>
                    <select id="order_month">
                        <option value="1">January</option><option value="2">February</option>
                        <option value="3">March</option><option value="4">April</option>
                        <option value="5">May</option><option value="6">June</option>
                        <option value="7">July</option><option value="8">August</option>
                        <option value="9">September</option><option value="10">October</option>
                        <option value="11">November</option><option value="12">December</option>
                    </select>
                </div>
            </div>
            <button class="btn" onclick="predict()">⚡ Predict Now</button>
            <div id="result" style="display:none;"></div>
        </div>
    </div>
    <script>
        function predict() {
            const mode = document.getElementById('shipping_mode').value;
            const priority = document.getElementById('order_priority').value;
            const distance = parseInt(document.getElementById('distance').value);
            const quantity = parseInt(document.getElementById('quantity').value);
            const month = parseInt(document.getElementById('order_month').value);

            // Simple rule-based prediction (mirrors the ML model logic)
            let riskScore = 0;

            // Distance risk
            if (distance > 1500) riskScore += 40;
            else if (distance > 800) riskScore += 20;
            else if (distance > 300) riskScore += 5;

            // Shipping mode risk
            if (mode === 'Economy') riskScore += 30;
            else if (mode === 'Standard') riskScore += 15;
            else if (mode === 'Same Day') riskScore += 25; // operational stress
            else riskScore += 5; // Express

            // Priority risk (low priority = more delays)
            if (priority === 'Low') riskScore += 10;
            else if (priority === 'Critical') riskScore -= 5;

            // Peak season
            if ([10, 11, 12].includes(month)) riskScore += 10;

            const isDelayed = riskScore >= 45;
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';

            if (isDelayed) {
                resultDiv.className = 'result delayed';
                resultDiv.innerHTML = `
                    <div class="result-icon">⚠️</div>
                    <div class="result-title">DELAYED</div>
                    <div class="result-subtitle">This order is likely to be <strong>delayed</strong> based on the given parameters.<br>
                    Consider using Express shipping or reducing distance risk.</div>`;
            } else {
                resultDiv.className = 'result ontime';
                resultDiv.innerHTML = `
                    <div class="result-icon">✅</div>
                    <div class="result-title">ON TIME</div>
                    <div class="result-subtitle">This order is predicted to be delivered <strong>on time</strong>.<br>
                    Current parameters are within acceptable delivery thresholds.</div>`;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    shipping_mode = data.get('shipping_mode', 'Standard')
    distance = float(data.get('distance', 500))
    priority = data.get('priority', 'Medium')
    month = int(data.get('month', 6))

    risk = 0
    if distance > 1500: risk += 40
    elif distance > 800: risk += 20
    elif distance > 300: risk += 5
    if shipping_mode == 'Economy': risk += 30
    elif shipping_mode == 'Standard': risk += 15
    elif shipping_mode == 'Same Day': risk += 25
    if priority == 'Low': risk += 10
    if month in [10, 11, 12]: risk += 10

    prediction = 'Delayed' if risk >= 45 else 'On Time'
    return jsonify({'prediction': prediction, 'risk_score': risk})

if __name__ == '__main__':
    print("Starting Supply Chain Delay Predictor...")
    print("Open browser: http://localhost:5000")
    app.run(debug=True, port=5000)
