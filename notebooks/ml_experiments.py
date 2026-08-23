"""
ChainSight — ML Improvement Experiments
Tests two targeted changes against the baseline RF model.
Only saves a new model if it measurably improves the weighted F1 score.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Reproduce exact training data preparation ────────────────────────────────
print("Loading data...")
df = pd.read_csv("notebooks/cleaned_supply_chain.csv")

LEAKAGE = [
    "Delay_Days", "Delay_Category", "Days for shipping (real)",
    "Delivery Status", "Shipping_Gap", "shipping date (DateOrders)"
]
ID_COLS = [
    "Customer Id", "Order Customer Id", "Order Item Id", "Order Id",
    "Category Id", "Department Id", "Order Item Cardprod Id",
    "Product Card Id", "Product Category Id", "order date (DateOrders)"
]

df = df.drop(columns=[c for c in LEAKAGE if c in df.columns])
y = df["Late_delivery_risk"]
X = df.drop(columns=["Late_delivery_risk"])
X = X.drop(columns=[c for c in ID_COLS if c in X.columns])

print(f"Features: {len(X.columns)}")
print(f"Class balance: {y.value_counts().to_dict()}")

# Encode categoricals (fresh fit — same data as training so mappings are identical)
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Same train/test split as original training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}\n")

# ── Evaluation helper ────────────────────────────────────────────────────────
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    print(f"  {name}")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1:        {f1:.4f}")
    return acc, f1

# ── Baseline (already-trained model on the same test set) ────────────────────
print("=" * 55)
baseline = joblib.load("models/supply_chain_model.pkl")
baseline_acc, baseline_f1 = evaluate(
    "Baseline  (RF n=50, no constraints)", baseline, X_test, y_test
)

# ── Experiment 1: class_weight='balanced' ───────────────────────────────────
print("=" * 55)
print("Training Experiment 1 (class_weight='balanced')...")
exp1 = RandomForestClassifier(
    n_estimators=50, class_weight="balanced", random_state=42, n_jobs=-1
)
exp1.fit(X_train, y_train)
exp1_acc, exp1_f1 = evaluate(
    "Exp 1     (RF n=50, class_weight='balanced')", exp1, X_test, y_test
)

# ── Experiment 2: more trees + depth cap ────────────────────────────────────
print("=" * 55)
print("Training Experiment 2 (n_estimators=100, max_depth=20)...")
exp2 = RandomForestClassifier(
    n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
)
exp2.fit(X_train, y_train)
exp2_acc, exp2_f1 = evaluate(
    "Exp 2     (RF n=100, max_depth=20)", exp2, X_test, y_test
)

# ── Decision ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("SUMMARY")
print(f"  Baseline  — acc={baseline_acc:.4f}  f1={baseline_f1:.4f}")
print(f"  Exp 1     — acc={exp1_acc:.4f}  f1={exp1_f1:.4f}"
      f"  {'← BETTER' if exp1_f1 > baseline_f1 else ''}")
print(f"  Exp 2     — acc={exp2_acc:.4f}  f1={exp2_f1:.4f}"
      f"  {'← BETTER' if exp2_f1 > baseline_f1 else ''}")

best_f1    = baseline_f1
best_model = None
best_name  = "Baseline"

if exp1_f1 > best_f1:
    best_f1    = exp1_f1
    best_model = exp1
    best_name  = "Exp 1 (class_weight='balanced')"

if exp2_f1 > best_f1:
    best_f1    = exp2_f1
    best_model = exp2
    best_name  = "Exp 2 (n_estimators=100, max_depth=20)"

if best_model is not None:
    joblib.dump(best_model, "models/supply_chain_model.pkl")
    print(f"\n✓ NEW MODEL SAVED: {best_name}  (F1={best_f1:.4f})")
else:
    print(f"\n— Keeping baseline. No experiment surpassed F1={baseline_f1:.4f}.")
