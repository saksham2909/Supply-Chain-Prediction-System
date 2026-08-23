# ChainSight — GitHub / Resume Presentation

## Project Description

ChainSight is an end-to-end machine learning system that predicts shipment delivery delays in a global supply chain. It takes a raw DataCo order export (CSV) and scores every row in real time — returning a `Late / On Time` prediction and a 0–100% risk score for each order, before dispatch.

The system includes a preprocessing pipeline that removes data-leakage columns, engineers time-based features, and aligns inputs to the trained model's expected format. A Flask API handles the file upload and prediction, and a fully custom dark-themed dashboard renders the results with interactive visualizations.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML model** | scikit-learn `RandomForestClassifier` |
| **Data processing** | pandas, scikit-learn `LabelEncoder` |
| **Backend** | Flask (Python) |
| **Model persistence** | joblib |
| **Frontend** | HTML5, Tailwind CSS, Vanilla JavaScript |
| **Dataset** | DataCo Smart Supply Chain (Kaggle) |

---

## Measurable Achievements

> All numbers are verified from actual model evaluation on the held-out test set (36,104 rows).

- **76.15% accuracy** on a 180,519-row real-world supply chain dataset
- **84.95% precision** — when the model predicts Late, it is correct 85% of the time
- **75.95% weighted F1** across an imbalanced class distribution (55% Late / 45% On-Time)
- Processes the **full 180K+ row dataset** end-to-end through the Flask API (preprocessing + inference)
- **33 features** used per prediction, engineered from raw order, customer, product, and logistics fields
- Two ML experiments were run (`class_weight='balanced'` and `n_estimators=100, max_depth=20`); the baseline outperformed both — results are documented in `notebooks/ml_experiments.py`

---

## What Makes This Project Interesting

1. **Leakage-safe pipeline** — columns like `Late_delivery_risk`, `Days for shipping (real)`, and `Delivery Status` would trivially leak the target. The preprocessing pipeline strips these at inference time, and the model was trained without them.

2. **Production-like structure** — the project separates the preprocessing logic (`preprocessing/preprocessing.py`) from the Flask app (`app.py`), so the same pipeline code is used for both training-time preparation and inference.

3. **Honest ML experimentation** — two targeted experiments were run and documented. Neither beat the baseline, and the baseline was kept without overclaiming an improvement.

4. **Handles large files** — tested against the full 180K+ row dataset through the HTTP API, not just a small sample.

---

## Sample Project Description (for LinkedIn / CV)

> **ChainSight — Supply Chain Delay Predictor**
> Built an end-to-end ML system to predict delivery delays on 180K+ real supply chain orders (DataCo dataset). Trained a Random Forest classifier achieving 76% accuracy and 85% precision. Built a Flask REST API with a preprocessing pipeline that removes data leakage, engineers time-based features, and scores each shipment before dispatch. Frontend dashboard renders live Late/On-Time splits, risk breakdowns, and a top-10 highest-risk orders view — with CSV export.
> Tech: Python · Flask · scikit-learn · pandas · JavaScript

---

## Bullet Points (resume format)

- Trained a Random Forest model on **180K+ real supply chain orders** achieving **76% accuracy** and **85% precision** on a held-out test set
- Built a **Flask REST API** that accepts CSV uploads, runs a leakage-safe preprocessing pipeline, and returns per-order Late/On-Time predictions with confidence scores
- Engineered **33 features** from raw DataCo fields including shipping mode, order timing, product category, and geographic signals
- Designed an interactive **dark-theme dashboard** with Late vs On-Time stacked visualization, risk tier breakdown, Top 10 highest-risk orders, and CSV export
- Ran **two controlled ML experiments** (`class_weight='balanced'`, `max_depth=20`) against the baseline, documented results, and kept the baseline when neither improved F1
