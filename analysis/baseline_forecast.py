import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# STEP 1: SET PROJECT PATHS
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

file_path = project_folder / "data" / "train.csv"
output_path = project_folder / "analysis" / "baseline_forecast_results.csv"
chart_path = project_folder / "analysis" / "baseline_forecast.png"


# ============================================================
# STEP 2: LOAD THE DATASET
# ============================================================

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print(f"Total records: {len(df)}")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])


# ============================================================
# STEP 3: PREPARE DAILY SALES DATA
# ============================================================

# Combine sales from all stores and product categories
daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .sort_index()
)

# Ensure every calendar day is represented
daily_sales = daily_sales.asfreq("D")

print(f"\nTotal number of days: {len(daily_sales)}")
print(f"Start date: {daily_sales.index.min().date()}")
print(f"End date: {daily_sales.index.max().date()}")

# Check for missing daily sales values
missing_days = daily_sales.isna().sum()

print(f"Missing daily sales values: {missing_days}")

# Handle missing daily values through time interpolation
if missing_days > 0:
    print("Filling missing daily sales values using time interpolation...")

    daily_sales = daily_sales.interpolate(method="time")

# Check whether any missing values remain
remaining_missing = daily_sales.isna().sum()

print(f"Missing values after handling: {remaining_missing}")

if remaining_missing > 0:
    raise ValueError(
        "Missing values remain after interpolation. "
        "Please investigate the dataset before forecasting."
    )


# ============================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

# Use the final 30 days for testing
test_days = 30

if len(daily_sales) <= test_days:
    raise ValueError(
        "The dataset must contain more than 30 days of sales data."
    )

train = daily_sales.iloc[:-test_days].copy()
test = daily_sales.iloc[-test_days:].copy()

print("\nTraining and testing split completed.")
print(f"Training days: {len(train)}")
print(f"Testing days: {len(test)}")

print(f"Training period: {train.index.min().date()} to "
      f"{train.index.max().date()}")

print(f"Testing period: {test.index.min().date()} to "
      f"{test.index.max().date()}")


# ============================================================
# STEP 5: CREATE A BASELINE FORECAST
# ============================================================

# Seasonal naive forecasting:
# Predict each day using the value from seven days earlier.
#
# For dates beyond the training period, use previously generated
# predictions rather than actual test values.

history = train.copy()

predictions = []

for date in test.index:

    source_date = date - pd.Timedelta(days=7)

    if source_date not in history.index:
        raise ValueError(
            f"Cannot find a value for the forecast date {source_date.date()}."
        )

    prediction = history.loc[source_date]

    predictions.append(prediction)

    # Add the prediction to history so future forecast days
    # can use it without accessing actual test values.
    history.loc[date] = prediction

forecast = pd.Series(
    predictions,
    index=test.index,
    name="Predicted_Sales"
)

print("\nBaseline forecasting completed!")


# ============================================================
# STEP 6: EVALUATE THE FORECAST
# ============================================================

# Mean Absolute Error
mae = mean_absolute_error(test, forecast)

# Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(test, forecast))

# Weighted Absolute Percentage Error
total_actual_sales = np.abs(test).sum()

if total_actual_sales != 0:
    wape = (
        np.abs(test - forecast).sum()
        / total_actual_sales
    ) * 100
else:
    wape = np.nan

print("\n========== BASELINE FORECAST RESULTS ==========")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

if np.isnan(wape):
    print("WAPE : Cannot calculate because total actual sales is zero.")
else:
    print(f"WAPE : {wape:.2f}%")

print("================================================")


# ============================================================
# STEP 7: SAVE FORECAST RESULTS
# ============================================================

results = pd.DataFrame({
    "Date": test.index,
    "Actual_Sales": test.values,
    "Predicted_Sales": forecast.values
})

results["Absolute_Error"] = np.abs(
    results["Actual_Sales"] - results["Predicted_Sales"]
)

results.to_csv(output_path, index=False)

print(f"\nForecast results saved to:\n{output_path}")


# ============================================================
# STEP 8: VISUALIZE ACTUAL VS PREDICTED SALES
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    test.index,
    test.values,
    label="Actual Sales",
    linewidth=2
)

plt.plot(
    forecast.index,
    forecast.values,
    label="Baseline Forecast",
    linestyle="--",
    linewidth=2
)

plt.title("SalesSense AI - Baseline Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Total Daily Sales")

plt.legend()
plt.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(chart_path, dpi=300, bbox_inches="tight")

print(f"Forecast chart saved to:\n{chart_path}")

plt.show()


# ============================================================
# STEP 9: DISPLAY FIRST 10 FORECAST RESULTS
# ============================================================

print("\nFirst 10 forecast predictions:")
print(results.head(10).to_string(index=False))

print("\nBaseline forecasting completed successfully!")