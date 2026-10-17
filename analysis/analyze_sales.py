import pandas as pd
from pathlib import Path

# Locate and load the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "train.csv"

df = pd.read_csv(file_path)

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

print("\n===== SALES DATASET OVERVIEW =====")
print("Total records:", len(df))
print("Total stores:", df["store_nbr"].nunique())
print("Product categories:", df["family"].nunique())
print("Start date:", df["date"].min().date())
print("End date:", df["date"].max().date())

print("\n===== SALES STATISTICS =====")
print(df["sales"].describe())

print("\n===== TOTAL SALES BY PRODUCT CATEGORY =====")
category_sales = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print(category_sales.head(10))

print("\n===== TOTAL SALES BY STORE =====")
store_sales = (
    df.groupby("store_nbr")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print(store_sales.head(10))

print("\n===== DAILY TOTAL SALES =====")
daily_sales = df.groupby("date")["sales"].sum()

print(daily_sales.head(10))

print("\nAnalysis completed successfully!")