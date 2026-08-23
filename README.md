# ChainSight — Supply Chain Delivery Delay Prediction

> Predict whether a shipment will arrive late — before it leaves the warehouse.

ChainSight is a machine-learning web application that scores every order in a DataCo supply chain CSV and returns a `Late / On Time` label plus a 0–100% risk score for each row. Built with Flask, scikit-learn, and a 180K+ order training dataset.

---

## Problem Statement

Late deliveries are a leading driver of customer churn and margin erosion in e-commerce supply chains. Most delay detection happens after the fact. ChainSight shifts that decision point **before** dispatch — giving operations teams time to act.

---

## Architecture

```
CSV Upload (browser)
      │
      ▼
Flask /predict endpoint (app.py)
      │
      ├── clean_data()        — drop leakage cols, engineer date features
      ├── encode_data()       — LabelEncoder transform (33 categorical → numeric)
      ├── prepare_for_model() — align columns to model.feature_names_in_
      │
      ▼
RandomForestClassifier  ←  models/supply_chain_model.pkl
      │
      ▼
JSON: [{prediction, risk}, ...]
      │
      ▼
Frontend dashboard (index.html)
  • Late vs On-Time stacked bar
  • Low / Medium / High risk breakdown
  • Top 10 highest-risk orders
  • Paginated results table
  • CSV export
```

---

## Dataset

**DataCo Smart Supply Chain for Big Data Analysis** — sourced from Kaggle.

- **Rows**: 180,519 orders  
- **Features used**: 33 signals (after dropping leakage and PII columns)  
- **Target**: `Late_delivery_risk` (1 = Late, 0 = On Time)  
- **Class split**: ~55% Late / ~45% On Time

### Key Features

| Category | Features |
|---|---|
| Order timing | Days for shipment (scheduled), Month, Peak_Season, Order_Day, Is_Weekend |
| Product | Product Name, Product Price, Category Name, Department Name |
| Customer | Customer Segment, Customer City, Customer Country, Customer State |
| Geography | Market, Order Region, Order Country, Order City, Latitude, Longitude |
| Financials | Sales, Order Item Total, Benefit per order, Order Profit Per Order, Order Item Profit Ratio |
| Logistics | Shipping Mode, Order_Value_Category |

---

## Model

**Algorithm**: Random Forest Classifier  
**Library**: scikit-learn  
**Training split**: 80/20 stratified (random_state=42)

### Evaluation Results (test set, 36,104 rows)

| Metric | Score |
|---|---|
| **Accuracy** | **76.15%** |
| **Precision** | **84.95%** |
| **Recall** | **68.68%** |
| **F1 Score** | **75.95%** |

### Experiments Tried

Two experiments were run against the baseline. Neither improved on the baseline F1, so the original model is kept.

| Experiment | Accuracy | F1 | Outcome |
|---|---|---|---|
| Baseline (RF, n=50) | 76.15% | 75.95% | **Kept** |
| class_weight='balanced' | 75.82% | 75.70% | — |
| n_estimators=100, max_depth=20 | 71.88% | 69.77% | — |

> Note: `class_weight='balanced'` slightly hurts precision with only marginal recall gain. `max_depth=20` under-fits the data. The default unlimited depth works best here.

---

## How to Run

### Prerequisites

```bash
python -m pip install flask pandas scikit-learn joblib
```

### Start the server

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### Upload a CSV

The CSV must contain the original DataCo column names. Leakage columns (`Late_delivery_risk`, `Days for shipping (real)`, `Delivery Status`, etc.) are automatically stripped if present — they won't contaminate predictions.

### Output

Each row in the response has:
```json
{ "prediction": "Late" | "On Time", "risk": 0.0 - 100.0 }
```

---

## Project Structure

```
Supply Chain Delay Prediction System/
├── app.py                          # Flask app — /predict endpoint
├── preprocessing/
│   └── preprocessing.py            # clean_data, encode_data, prepare_for_model
├── models/
│   ├── supply_chain_model.pkl      # Trained RF model (~181 MB)
│   └── encoders.pkl                # LabelEncoders for categorical columns
├── templates/
│   └── index.html                  # Dark-theme dashboard UI
├── notebooks/
│   ├── data_cleaning.ipynb         # EDA + feature engineering
│   ├── model_training.py           # Original training script
│   └── ml_experiments.py          # Hyperparameter experiments
└── README.md
```

---

## Tech Stack

- **Python 3** — core language
- **Flask** — lightweight REST backend
- **pandas** — CSV ingestion and preprocessing
- **scikit-learn** — RandomForestClassifier, LabelEncoder, train/test split
- **joblib** — model serialization
- **HTML / Tailwind CSS / Vanilla JS** — frontend dashboard

---

## Limitations & Next Steps

- The model was trained on DataCo data; CSVs with unseen categorical values (e.g., new country names) will return a 400 error — this is intentional, not a bug.
- Precision (84.95%) is strong; Recall (68.68%) could improve with more training data or ensemble methods.
- A future improvement would be to retrain the encoders with `handle_unknown='ignore'` semantics to gracefully handle novel categories.
