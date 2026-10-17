import pandas as pd
from pathlib import Path

# Locate the sales dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "train.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Display basic information
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATASET SHAPE ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- DATA TYPES AND MISSING VALUES ---")
df.info()

print("\n--- MISSING VALUES PER COLUMN ---")
print(df.isnull().sum())