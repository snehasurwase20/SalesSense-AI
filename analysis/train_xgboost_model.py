import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from xgboost import XGBRegressor


# =====================================================
# STEP 1: SET PROJECT PATHS
# =====================================================

project_folder = Path(__file__).resolve().parent.parent

data_path = project_folder / "data" / "train.csv"
model_folder = project_folder / "models"

model_path = model_folder / "xgboost_sales_model.json"
metadata_path = model_folder / "xgboost_model_metadata.joblib"


# Create models folder if it does not exist
model_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# STEP 2: LOAD DATA
# =====================================================

df = pd.read_csv(data_path)

df["date"] = pd.to_datetime(df["date"])

daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .sort_index()
)

daily_sales = daily_sales.asfreq("D")

missing_days = daily_sales.isna().sum()

print("Dataset loaded successfully!")
print(f"Total days: {len(daily_sales)}")
print(f"Missing daily values: {missing_days}")

if missing_days > 0:
    daily_sales = daily_sales.interpolate(method="time")

if daily_sales.isna().any():
    raise ValueError("Missing sales values remain after interpolation.")


# =====================================================
# STEP 3: CREATE FEATURES
# =====================================================

data = pd.DataFrame({
    "date": daily_sales.index,
    "sales": daily_sales.values
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

# Historical sales features
data["lag_1"] = data["sales"].shift(1)
data["lag_7"] = data["sales"].shift(7)
data["lag_14"] = data["sales"].shift(14)
data["lag_28"] = data["sales"].shift(28)

# Rolling averages based only on previous days
data["rolling_mean_7"] = (
    data["sales"].shift(1).rolling(7).mean()
)

data["rolling_mean_14"] = (
    data["sales"].shift(1).rolling(14).mean()
)

data["rolling_mean_28"] = (
    data["sales"].shift(1).rolling(28).mean()
)

data = data.dropna().reset_index(drop=True)


# =====================================================
# STEP 4: PREPARE TRAINING DATA
# =====================================================

feature_columns = [
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
    "rolling_mean_28",
]

# Keep the final 30 days out of model training.
# This preserves the test period used in our comparison.
test_days = 30

if len(data) <= test_days:
    raise ValueError("Not enough data for the selected test period.")

train = data.iloc[:-test_days].copy()

X_train = train[feature_columns]
y_train = train["sales"]

print(f"\nTraining records: {len(train)}")
print(f"Excluded test records: {test_days}")


# =====================================================
# STEP 5: TRAIN XGBOOST MODEL
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

print("Training completed!")


# =====================================================
# STEP 6: SAVE THE TRAINED MODEL
# =====================================================

model.save_model(str(model_path))

# Save the latest observed daily sales and feature names.
# The API will use these to build future prediction features.
metadata = {
    "feature_columns": feature_columns,
    "last_training_date": train["date"].iloc[-1],
    "last_observed_sales": daily_sales.copy(),
    "test_days_excluded": test_days,
}

joblib.dump(metadata, metadata_path)


# =====================================================
# STEP 7: CONFIRM SAVED FILES
# =====================================================

print("\n========== MODEL SAVING RESULTS ==========")

print(f"Model saved to:\n{model_path}")
print(f"\nMetadata saved to:\n{metadata_path}")

print("\nModel files created successfully!")