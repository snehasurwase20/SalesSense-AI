import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =====================================================
# STEP 1: PROJECT PATHS
# =====================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_FOLDER / "data" / "train.csv"
MODELS_FOLDER = PROJECT_FOLDER / "models"

MODEL_PATH = MODELS_FOLDER / "xgboost_sales_model.json"
METADATA_PATH = MODELS_FOLDER / "xgboost_model_metadata.joblib"

RESULTS_PATH = PROJECT_FOLDER / "analysis" / "xgboost_forecast_results.csv"
CHART_PATH = PROJECT_FOLDER / "analysis" / "xgboost_forecast.png"

MODELS_FOLDER.mkdir(parents=True, exist_ok=True)


# =====================================================
# STEP 2: LOAD DATA
# =====================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
df = df.dropna(subset=["date", "sales"])

daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .sort_index()
    .asfreq("D")
)

print("Dataset loaded successfully!")
print(f"Total days: {len(daily_sales)}")

missing_days = daily_sales.isna().sum()
print(f"Missing daily values: {missing_days}")

if missing_days:
    daily_sales = daily_sales.interpolate(method="time")

if daily_sales.isna().any():
    raise ValueError("Missing sales values remain after interpolation.")

if len(daily_sales) < 100:
    raise ValueError("Not enough data to train the model.")


# =====================================================
# STEP 3: FEATURE ENGINEERING
# =====================================================

def create_features(sales):
    data = pd.DataFrame({
        "date": sales.index,
        "sales": sales.values
    })

    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    data["year"] = data["date"].dt.year
    data["day_of_year"] = data["date"].dt.dayofyear
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["lag_1"] = data["sales"].shift(1)
    data["lag_7"] = data["sales"].shift(7)
    data["lag_14"] = data["sales"].shift(14)
    data["lag_28"] = data["sales"].shift(28)

    data["rolling_mean_7"] = (
        data["sales"].shift(1).rolling(7).mean()
    )

    data["rolling_mean_14"] = (
        data["sales"].shift(1).rolling(14).mean()
    )

    data["rolling_mean_28"] = (
        data["sales"].shift(1).rolling(28).mean()
    )

    return data


FEATURE_COLUMNS = [
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "year",
    "day_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28"
]


# =====================================================
# STEP 4: TRAIN-TEST SPLIT
# =====================================================

data = create_features(daily_sales)
data = data.dropna().reset_index(drop=True)

test_days = 30

if len(data) <= test_days:
    raise ValueError("Not enough data for testing.")

train = data.iloc[:-test_days].copy()
test = data.iloc[-test_days:].copy()

X_train = train[FEATURE_COLUMNS]
y_train = train["sales"]

X_test = test[FEATURE_COLUMNS]
y_test = test["sales"]

print("\nTraining and testing split completed.")
print(f"Training rows: {len(train)}")
print(f"Testing rows: {len(test)}")


# =====================================================
# STEP 5: TRAIN XGBOOST
# =====================================================

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost model...")

model.fit(X_train, y_train)

print("Model training completed!")


# =====================================================
# STEP 6: FORECAST THE TEST PERIOD
# =====================================================

# Start with sales observed before the test period.
history = daily_sales.loc[
    daily_sales.index < pd.Timestamp(test["date"].iloc[0])
].copy()

predictions = []

for date_value in test["date"]:

    date = pd.Timestamp(date_value)

    row = {
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "month": date.month,
        "quarter": date.quarter,
        "year": date.year,
        "day_of_year": date.dayofyear,
        "is_weekend": int(date.dayofweek >= 5),

        "lag_1": float(history.iloc[-1]),
        "lag_7": float(history.iloc[-7]),
        "lag_14": float(history.iloc[-14]),
        "lag_28": float(history.iloc[-28]),

        "rolling_mean_7": float(history.iloc[-7:].mean()),
        "rolling_mean_14": float(history.iloc[-14:].mean()),
        "rolling_mean_28": float(history.iloc[-28:].mean())
    }

    X_future = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS
    )

    prediction = float(model.predict(X_future)[0])
    prediction = max(0.0, prediction)

    predictions.append(prediction)

    # Add the prediction to history for the next forecast.
    history.loc[date] = prediction

forecast = np.array(predictions)


# =====================================================
# STEP 7: EVALUATE MODEL
# =====================================================

mae = mean_absolute_error(y_test, forecast)

rmse = np.sqrt(
    mean_squared_error(y_test, forecast)
)

total_actual_sales = np.abs(y_test.to_numpy()).sum()

if total_actual_sales != 0:
    wape = (
        np.abs(y_test.to_numpy() - forecast).sum()
        / total_actual_sales
    ) * 100
else:
    wape = np.nan

print("\n========== XGBOOST RESULTS ==========")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

if np.isfinite(wape):
    print(f"WAPE : {wape:.2f}%")
else:
    print("WAPE : Not available")

print("=====================================")


# =====================================================
# STEP 8: SAVE TRAINED MODEL
# =====================================================

model.save_model(str(MODEL_PATH))

metadata = {
    "feature_columns": FEATURE_COLUMNS,
    "model_name": "XGBoost",
    "target_column": "sales",
    "test_days": test_days,
    "mae": float(mae),
    "rmse": float(rmse),
    "wape": float(wape) if np.isfinite(wape) else None
}

joblib.dump(metadata, METADATA_PATH)

print("\nModel saved successfully!")
print(f"Model: {MODEL_PATH}")
print(f"Metadata: {METADATA_PATH}")


# =====================================================
# STEP 9: SAVE FORECAST RESULTS
# =====================================================

results = pd.DataFrame({
    "Date": pd.to_datetime(test["date"]).to_numpy(),
    "Actual_Sales": y_test.to_numpy(),
    "Predicted_Sales": forecast
})

results["Absolute_Error"] = np.abs(
    results["Actual_Sales"] -
    results["Predicted_Sales"]
)

results.to_csv(RESULTS_PATH, index=False)

print(f"\nResults saved to: {RESULTS_PATH}")


# =====================================================
# STEP 10: PLOT ACTUAL VS PREDICTED SALES
# =====================================================

plt.figure(figsize=(14, 6))

plt.plot(
    test["date"],
    y_test.to_numpy(),
    label="Actual Sales",
    linewidth=2
)

plt.plot(
    test["date"],
    forecast,
    label="XGBoost Forecast",
    linestyle="--",
    linewidth=2
)

plt.title("SalesSense AI - XGBoost Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Total Daily Sales")

plt.legend()
plt.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    CHART_PATH,
    dpi=300,
    bbox_inches="tight"
)

print(f"Chart saved to: {CHART_PATH}")

plt.show()


# =====================================================
# STEP 11: DISPLAY SAMPLE RESULTS
# =====================================================

print("\nFirst 10 predictions:")
print(results.head(10).to_string(index=False))

print("\nXGBoost forecasting completed successfully!")