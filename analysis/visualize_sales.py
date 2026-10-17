import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate the project and dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "train.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

# Calculate total sales for each day
daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .sort_index()
)

# Plot daily sales
plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales.index,
    daily_sales.values,
    linewidth=1
)

plt.title("Daily Retail Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the chart
output_path = project_folder / "analysis" / "daily_sales_trend.png"

plt.savefig(output_path, dpi=300)
plt.show()

print("Chart saved successfully!")
print("Location:", output_path)