import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# cleaned dataset load kr rhe h

df = pd.read_csv("notebooks/cleaned_supply_chain.csv")

# leakage column remove kr rhe h

df = df.drop(columns=[
    "Delay_Days",
    "Delay_Category",
    "Days for shipping (real)",
    "Delivery Status",
    "Shipping_Gap",
    "shipping date (DateOrders)"
])

# target alag kr rhe h

X = df.drop(columns=["Late_delivery_risk"])

# useless ID columns remove kr rhe h

X = X.drop(columns=[
    "Customer Id",
    "Order Customer Id",
    "Order Item Id",
    "Order Id",
    "Category Id",
    "Department Id",
    "Order Item Cardprod Id",
    "Product Card Id",
    "Product Category Id",
    "order date (DateOrders)"
])

y = df["Late_delivery_risk"]

# categorical columns ko label encode kr rhe h

categorical_cols = X.select_dtypes(include="object").columns

encoders = {}

for col in categorical_cols:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    encoders[col] = encoder

print("Total features:", len(X.columns))
print(X.columns.tolist())

# data ko training aur testing me divide kr rhe h

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)





# Model training yha se shuru h

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("Model training start")

# logistic regression model bana rhe h

logistic_model = LogisticRegression(max_iter=2000)
logistic_model.fit(X_train, y_train)

# random forest model bana rhe h

rf_model = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

rf_model.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier

# decision tree model bana rhe h

dt_model = DecisionTreeClassifier(
    max_depth=20,
    random_state=42
)

dt_model.fit(X_train, y_train)

print("Decision Tree trained successfully")

print("Models trained successfully")

print("Decision Tree Train Accuracy:", dt_model.score(X_train, y_train))
print("Decision Tree Test Accuracy:", dt_model.score(X_test, y_test))

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# test data par prediction kr rhe h

logistic_pred = logistic_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# logistic regression ka result

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, logistic_pred))
print("Precision:", precision_score(y_test, logistic_pred))
print("Recall:", recall_score(y_test, logistic_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))

# random forest ka result

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# decision tree ki prediction kr rhe h

dt_pred = dt_model.predict(X_test)

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred))
print("Recall:", recall_score(y_test, dt_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

# random forest ke important features dekh rhe h

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

print(importance.sort_values(ascending=False).head(15))



#Saving the models using joblib
import joblib
import os

# Create a directory to save the models if it doesn't exist
os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/supply_chain_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("Model and encoders saved successfully")