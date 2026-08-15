import pandas as pd


LEAKAGE_COLUMNS = [
    "Late_delivery_risk",
    "Delay_Days",
    "Delay_Category",
    "Days for shipping (real)",
    "Delivery Status",
    "Shipping_Gap",
    "shipping date (DateOrders)"
]

DROP_COLUMNS = [
    "Product Description",
    "Order Zipcode",
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Product Image",
    "Customer Zipcode",
    "Product Status",
    "Order Status"
]


def clean_data(df):
    df = df.copy()

    # leakage columns ko prediction se pehle remove karna
    df.drop(columns=LEAKAGE_COLUMNS, errors="ignore", inplace=True)

    # training notebook wali unnecessary columns
    df.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)

    # order date se required features banana
    if "order date (DateOrders)" not in df.columns:
        raise ValueError("Required column 'order date (DateOrders)' is missing.")

    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"]
    )

    df["Month"] = df["order date (DateOrders)"].dt.month

    peak_months = [1, 5, 7]
    df["Peak_Season"] = df["Month"].apply(
        lambda x: 1 if x in peak_months else 0
    )

    df["Order_Day"] = df["order date (DateOrders)"].dt.dayofweek
    df["Is_Weekend"] = df["Order_Day"].apply(
        lambda x: 1 if x >= 5 else 0
    )

    # Sales se order value category banana
    df["Order_Value_Category"] = pd.cut(
        df["Sales"],
        bins=[0, 100, 300, 1000, float("inf")],
        labels=["Low", "Medium", "High", "Very High"]
    )

    # raw order date model ko nahi deni
    df.drop(columns=["order date (DateOrders)"], inplace=True)

    return df


def encode_data(df, encoders):
    df = df.copy()

    for column, encoder in encoders.items():
        if column not in df.columns:
            continue

        try:
            df[column] = encoder.transform(df[column])
        except ValueError as e:
            raise ValueError(
                f"Unseen value found in column '{column}'. "
                f"Please use a value present in the training data."
            ) from e

    return df

def prepare_for_model(df, model):
    expected = list(model.feature_names_in_)

    missing = [col for col in expected if col not in df.columns]
    extra = [col for col in df.columns if col not in expected]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    df = df.drop(columns=extra)

    return df[expected]