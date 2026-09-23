import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


def generate_and_train():
    np.random.seed(42)
    days = 365
    dates = pd.date_range(start="2025-01-01", periods=days)

    # Simulate base demand with seasonal patterns
    base_demand = 50 + 15 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = np.random.normal(0, 5, days)
    sales = np.maximum(0, base_demand + noise).astype(int)

    df = pd.DataFrame({"date": dates, "demand": sales})

    # Feature Engineering
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Lag features (past values)
    df["lag_1"] = df["demand"].shift(1)
    df["lag_7"] = df["demand"].shift(7)
    df["lag_14"] = df["demand"].shift(14)
    df["lag_30"] = df["demand"].shift(30)

    # Drop rows containing NaN due to lag generation
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["lag_1", "lag_7", "lag_14", "lag_30", "day_of_week", "month"]
    X = df[feature_cols]
    y = df["demand"]

    # Temporal split: 80% train, 20% test (no random shuffling)
    split_idx = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

    # Train Regressor
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    # Calculate WAPE on test set
    preds = model.predict(X_test)
    wape = np.sum(np.abs(y_test - preds)) / np.sum(y_test)
    print(f"Model Training Complete. Test WAPE: {wape:.2%}")

    # Ensure artifacts directory exists and export model
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")
    print("Model successfully serialized to artifacts/model.joblib")


if __name__ == "__main__":
    generate_and_train()
