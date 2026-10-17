import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "train.csv"

# Load data
df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"])

# Calculate total sales by week
weekly_sales = (
    df.set_index("date")["sales"]
    .resample("W")
    .sum()
)

print("\n===== WEEKLY SALES =====")
print(weekly_sales.head(10))

print("\n===== WEEKLY SALES SUMMARY =====")
print(weekly_sales.describe())

# Plot weekly sales
plt.figure(figsize=(14, 6))
plt.plot(weekly_sales.index, weekly_sales.values)

plt.title("Weekly Retail Sales Trend")
plt.xlabel("Week")
plt.ylabel("Total Sales")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save chart
output_path = project_folder / "analysis" / "weekly_sales_trend.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("\nWeekly analysis completed!")
print("Chart saved at:", output_path)