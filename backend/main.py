from pathlib import Path
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

# Paths (repository root is the parent of backend/)
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_FOLDER / "data" / "daily_sales.csv"
MODEL_PATH = PROJECT_FOLDER / "models" / "xgboost_sales_model.json"
METADATA_PATH = PROJECT_FOLDER / "models" / "xgboost_model_metadata.joblib"
COMPARISON_PATH = PROJECT_FOLDER / "analysis" / "model_comparison_results.csv"

app = FastAPI(
    title="SalesSense AI",
    description="AI-Powered Sales Forecasting & Trend Intelligence System",
    version="1.3.0",
)

# Add the deployed Vercel URL in Render as FRONTEND_URL after frontend deployment.
allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
if frontend_url:
    allowed_origins.append(frontend_url)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and metadata once when the service starts.
model = None
metadata = None
model_loading_error = None
try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    metadata = joblib.load(METADATA_PATH)
    print("SalesSense AI model loaded successfully.")
except Exception as exc:
    model = None
    model_loading_error = str(exc)
    print(f"Model loading error: {model_loading_error}")


class ForecastRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=30)


def load_daily_sales() -> pd.Series:
    """Load daily totals from the compact, pre-aggregated CSV."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Compact dataset not found: {DATA_PATH}. Ensure data/daily_sales.csv is committed."
        )
    df = pd.read_csv(DATA_PATH)
    if not {"date", "sales"}.issubset(df.columns):
        raise ValueError("daily_sales.csv must contain 'date' and 'sales' columns.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"])
    if df.empty:
        raise ValueError("No valid sales records found in daily_sales.csv.")
    daily_sales = df.groupby("date")["sales"].sum().sort_index().asfreq("D", fill_value=0)
    if len(daily_sales) < 28:
        raise ValueError("At least 28 days of historical sales are required.")
    return daily_sales


@app.get("/")
def home():
    return {
        "project": "SalesSense AI",
        "message": "Backend is running successfully!",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "model_error": model_loading_error,
        "dataset_available": DATA_PATH.is_file(),
    }


@app.get("/model-performance")
def model_performance():
    if not COMPARISON_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Model comparison results not found. Ensure analysis/model_comparison_results.csv is committed.",
        )
    try:
        comparison = pd.read_csv(COMPARISON_PATH)
        required = {"Model", "MAE", "RMSE", "WAPE_Percent"}
        if not required.issubset(comparison.columns):
            raise ValueError("Comparison CSV must contain Model, MAE, RMSE, and WAPE_Percent columns.")
        comparison = comparison.dropna(subset=["Model", "MAE", "RMSE", "WAPE_Percent"])
        if comparison.empty:
            raise ValueError("The comparison CSV contains no valid results.")
        results = [
            {
                "model": str(row["Model"]),
                "mae": round(float(row["MAE"]), 2),
                "rmse": round(float(row["RMSE"]), 2),
                "wape_percent": round(float(row["WAPE_Percent"]), 2),
            }
            for _, row in comparison.iterrows()
        ]
        baseline = next((x for x in results if x["model"].strip().lower() == "baseline seasonal naive"), None)
        xgb_result = next((x for x in results if x["model"].strip().lower() == "xgboost"), None)
        improvements = None
        if baseline and xgb_result:
            improvements = {
                "mae_improvement_percent": round((baseline["mae"] - xgb_result["mae"]) / baseline["mae"] * 100, 2) if baseline["mae"] else 0,
                "rmse_improvement_percent": round((baseline["rmse"] - xgb_result["rmse"]) / baseline["rmse"] * 100, 2) if baseline["rmse"] else 0,
                "wape_reduction_percentage_points": round(baseline["wape_percent"] - xgb_result["wape_percent"], 2),
            }
        return {
            "project": "SalesSense AI",
            "metrics_source": "analysis/model_comparison_results.csv",
            "note": "These are saved model evaluation results, not metrics recalculated during each forecast request.",
            "models": results,
            "improvements": improvements,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read model performance results: {exc}")


@app.get("/historical-analysis")
def historical_analysis():
    """Return monthly and weekday sales summaries from daily_sales.csv."""
    try:
        daily_sales = load_daily_sales()
        daily_df = daily_sales.rename("sales").reset_index()
        daily_df.columns = ["date", "sales"]

        monthly = daily_df.copy()
        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
        monthly_summary = monthly.groupby("month", as_index=False).agg(
            total_sales=("sales", "sum"), average_daily_sales=("sales", "mean")
        )
        monthly_summary["total_sales"] = monthly_summary["total_sales"].round(2)
        monthly_summary["average_daily_sales"] = monthly_summary["average_daily_sales"].round(2)

        weekday_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        weekday = daily_df.copy()
        weekday["weekday_number"] = weekday["date"].dt.dayofweek
        weekday_summary = weekday.groupby("weekday_number", as_index=False).agg(
            average_sales=("sales", "mean"), total_sales=("sales", "sum")
        ).sort_values("weekday_number")
        weekday_summary["day"] = weekday_summary["weekday_number"].map(weekday_names)
        weekday_summary["average_sales"] = weekday_summary["average_sales"].round(2)
        weekday_summary["total_sales"] = weekday_summary["total_sales"].round(2)
        if monthly_summary.empty or weekday_summary.empty:
            raise ValueError("Not enough historical data to calculate sales trends.")

        peak_month = monthly_summary.loc[monthly_summary["total_sales"].idxmax()]
        low_month = monthly_summary.loc[monthly_summary["total_sales"].idxmin()]
        peak_weekday = weekday_summary.loc[weekday_summary["average_sales"].idxmax()]
        low_weekday = weekday_summary.loc[weekday_summary["average_sales"].idxmin()]
        return {
            "project": "SalesSense AI",
            "data_source": "data/daily_sales.csv",
            "note": "Historical results are aggregated from the original training dataset. Sales values are dataset values and are not assumed to be INR.",
            "date_range": {
                "start": daily_df["date"].min().strftime("%Y-%m-%d"),
                "end": daily_df["date"].max().strftime("%Y-%m-%d"),
            },
            "overall": {
                "total_sales": round(float(daily_df["sales"].sum()), 2),
                "average_daily_sales": round(float(daily_df["sales"].mean()), 2),
                "highest_sales_day": {
                    "date": daily_df.loc[daily_df["sales"].idxmax(), "date"].strftime("%Y-%m-%d"),
                    "sales": round(float(daily_df["sales"].max()), 2),
                },
            },
            "monthly": monthly_summary.to_dict(orient="records"),
            "weekdays": weekday_summary[["day", "average_sales", "total_sales"]].to_dict(orient="records"),
            "insights": {
                "highest_total_sales_month": {"month": str(peak_month["month"]), "total_sales": round(float(peak_month["total_sales"]), 2)},
                "lowest_total_sales_month": {"month": str(low_month["month"]), "total_sales": round(float(low_month["total_sales"]), 2)},
                "highest_average_sales_weekday": {"day": str(peak_weekday["day"]), "average_sales": round(float(peak_weekday["average_sales"]), 2)},
                "lowest_average_sales_weekday": {"day": str(low_weekday["day"]), "average_sales": round(float(low_weekday["average_sales"]), 2)},
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Historical analysis failed: {exc}")


@app.post("/forecast")
def forecast_sales(request: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Forecast model unavailable: {model_loading_error}")
    try:
        feature_columns = metadata.get("feature_columns") if isinstance(metadata, dict) else getattr(metadata, "feature_columns", None)
        if not feature_columns:
            raise ValueError("Feature columns are missing from model metadata.")

        history = load_daily_sales()
        forecasts = []
        for _ in range(request.days):
            next_date = history.index[-1] + pd.Timedelta(days=1)
            row = {
                "day_of_week": next_date.dayofweek,
                "day_of_month": next_date.day,
                "month": next_date.month,
                "quarter": next_date.quarter,
                "year": next_date.year,
                "day_of_year": next_date.dayofyear,
                "is_weekend": int(next_date.dayofweek >= 5),
                "lag_1": history.iloc[-1],
                "lag_7": history.iloc[-7],
                "lag_14": history.iloc[-14],
                "lag_28": history.iloc[-28],
                "rolling_mean_7": history.iloc[-7:].mean(),
                "rolling_mean_14": history.iloc[-14:].mean(),
                "rolling_mean_28": history.iloc[-28:].mean(),
            }
            missing_features = [column for column in feature_columns if column not in row]
            if missing_features:
                raise ValueError(f"Model features do not match API features: {missing_features}")
            X_future = pd.DataFrame([[row[column] for column in feature_columns]], columns=feature_columns)
            prediction = max(0.0, float(model.predict(X_future)[0]))
            history.loc[next_date] = prediction
            forecasts.append({"date": next_date.strftime("%Y-%m-%d"), "predicted_sales": round(prediction, 2)})
        return {"project": "SalesSense AI", "model": "XGBoost", "forecast_days": request.days, "forecasts": forecasts}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {exc}")
