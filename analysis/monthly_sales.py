import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate and load the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "train.csv"

df = pd.read_csv(file_path)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Calculate monthly sales
monthly_sales = (
    df.set_index("date")["sales"]
    .resample("MS")
    .sum()
)

print("\n===== MONTHLY SALES =====")
print(monthly_sales)

# Plot monthly sales
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")

plt.title("Monthly Retail Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save chart
output_path = project_folder / "analysis" / "monthly_sales_trend.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("\nMonthly analysis completed!")
print("Chart saved at:", output_path)