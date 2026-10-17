import pandas as pd
from pathlib import Path

# Project paths
project_folder = Path(__file__).resolve().parent.parent

output_path = (
    project_folder / "analysis" / "model_comparison_results.csv"
)

# Results obtained from the two forecasting models
comparison = pd.DataFrame({
    "Model": [
        "Baseline Seasonal Naive",
        "XGBoost"
    ],
    "MAE": [
        60671.64,
        58687.31
    ],
    "RMSE": [
        91695.52,
        76792.63
    ],
    "WAPE_Percent": [
        7.28,
        7.05
    ]
})

# Calculate percentage improvements over the baseline
baseline = comparison.iloc[0]
xgboost = comparison.iloc[1]

mae_improvement = (
    (baseline["MAE"] - xgboost["MAE"])
    / baseline["MAE"]
) * 100

rmse_improvement = (
    (baseline["RMSE"] - xgboost["RMSE"])
    / baseline["RMSE"]
) * 100

wape_improvement = (
    (baseline["WAPE_Percent"] - xgboost["WAPE_Percent"])
    / baseline["WAPE_Percent"]
) * 100

# Save the comparison
comparison.to_csv(output_path, index=False)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

print("\n========== XGBOOST IMPROVEMENT ==========")
print(f"MAE improvement: {mae_improvement:.2f}%")
print(f"RMSE improvement: {rmse_improvement:.2f}%")
print(f"WAPE reduction: {wape_improvement:.2f}%")

print(f"\nComparison saved to:\n{output_path}")
print("\nModel comparison completed successfully!")