"use client";

import { useEffect, useMemo, useState } from "react";

type ForecastItem = {
  date: string;
  predicted_sales: number;
};

type ForecastResponse = {
  project?: string;
  model?: string;
  forecast_days?: number;
  forecasts?: ForecastItem[];
};

type PerformanceModel = {
  model: string;
  mae: number;
  rmse: number;
  wape_percent: number;
};

type PerformanceResponse = {
  models: PerformanceModel[];
  improvements?: {
    mae_improvement_percent: number;
    rmse_improvement_percent: number;
    wape_reduction_percentage_points: number;
  } | null;
  note?: string;
};

type MonthlyItem = {
  month: string;
  total_sales: number;
  average_daily_sales: number;
};

type WeekdayItem = {
  day: string;
  average_sales: number;
  total_sales: number;
};

type HistoricalResponse = {
  project?: string;
  data_source?: string;
  note?: string;
  date_range?: { start: string; end: string };
  overall?: {
    total_sales: number;
    average_daily_sales: number;
    highest_sales_day: { date: string; sales: number };
  };
  monthly?: MonthlyItem[];
  weekdays?: WeekdayItem[];
  insights?: {
    highest_total_sales_month: { month: string; total_sales: number };
    lowest_total_sales_month: { month: string; total_sales: number };
    highest_average_sales_weekday: { day: string; average_sales: number };
    lowest_average_sales_weekday: { day: string; average_sales: number };
  };
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const formatSales = (value: number) =>
  new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2,
  }).format(value);

const formatCompactSales = (value: number) =>
  new Intl.NumberFormat("en-IN", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);

const formatMonth = (month: string) => {
  const [year, monthNumber] = month.split("-");
  const date = new Date(Number(year), Number(monthNumber) - 1, 1);
  return date.toLocaleDateString("en-IN", { month: "short", year: "numeric" });
};

function SalesChart({ items }: { items: ForecastItem[] }) {
  if (!items.length) return null;

  const values = items.map((item) => Number(item.predicted_sales) || 0);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const range = maxValue - minValue || 1;

  const width = 900;
  const height = 300;
  const paddingX = 55;
  const paddingTop = 30;
  const paddingBottom = 48;
  const chartHeight = height - paddingTop - paddingBottom;
  const chartWidth = width - paddingX * 2;

  const points = values.map((value, index) => {
    const x =
      paddingX +
      (values.length === 1
        ? chartWidth / 2
        : (index / (values.length - 1)) * chartWidth);
    const y = paddingTop + ((maxValue - value) / range) * chartHeight;

    return { x, y, value, date: items[index].date };
  });

  const linePoints = points.map((point) => `${point.x},${point.y}`).join(" ");
  const areaPoints = [
    `${paddingX},${height - paddingBottom}`,
    linePoints,
    `${width - paddingX},${height - paddingBottom}`,
  ].join(" ");

  const labelIndexes = Array.from(
    new Set([0, Math.floor((items.length - 1) / 2), items.length - 1]),
  );

  return (
    <div className="mt-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <span className="flex items-center gap-2 text-sm text-slate-600">
          <span className="h-3 w-3 rounded-full bg-teal-600" />
          Predicted sales
        </span>
        <span className="text-xs text-slate-500">
          {formatSales(minValue)} – {formatSales(maxValue)}
        </span>
      </div>
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Predicted daily sales line chart"
          className="h-auto w-full min-w-[500px]"
        >
          {[0, 1, 2, 3].map((step) => {
            const y = paddingTop + (chartHeight / 3) * step;
            const value = maxValue - (range / 3) * step;
            return (
              <g key={step}>
                <line
                  x1={paddingX}
                  y1={y}
                  x2={width - paddingX}
                  y2={y}
                  stroke="#e2e8f0"
                  strokeDasharray="4 5"
                />
                <text
                  x={paddingX - 10}
                  y={y + 4}
                  fill="#64748b"
                  fontSize="12"
                  textAnchor="end"
                >
                  {formatCompactSales(value)}
                </text>
              </g>
            );
          })}
          <polygon points={areaPoints} fill="#0d9488" opacity="0.10" />
          <polyline
            points={linePoints}
            fill="none"
            stroke="#0f766e"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {points.map((point) => (
            <g key={point.date}>
              <circle
                cx={point.x}
                cy={point.y}
                r="5"
                fill="white"
                stroke="#0f766e"
                strokeWidth="3"
              >
                <title>
                  {point.date}: {formatSales(point.value)}
                </title>
              </circle>
            </g>
          ))}
          {labelIndexes.map((index) => {
            const point = points[index];
            return (
              <text
                key={point.date}
                x={point.x}
                y={height - 15}
                fill="#64748b"
                fontSize="12"
                textAnchor={
                  index === 0
                    ? "start"
                    : index === items.length - 1
                      ? "end"
                      : "middle"
                }
              >
                {point.date}
              </text>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

function MonthlyChart({ items }: { items: MonthlyItem[] }) {
  if (!items.length) return null;
  const width = 1000;
  const height = 330;
  const padding = { left: 65, right: 20, top: 25, bottom: 65 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const maxValue = Math.max(...items.map((item) => item.total_sales), 1);
  const slotWidth = chartWidth / items.length;
  const barWidth = Math.max(2, slotWidth * 0.72);
  const labelStep = Math.max(1, Math.ceil(items.length / 12));

  return (
    <div className="w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Historical monthly total sales bar chart"
        className="h-auto w-full min-w-[650px]"
      >
        {[0, 1, 2, 3].map((step) => {
          const y = padding.top + (chartHeight / 3) * step;
          const value = maxValue - (maxValue / 3) * step;
          return (
            <g key={step}>
              <line
                x1={padding.left}
                y1={y}
                x2={width - padding.right}
                y2={y}
                stroke="#e2e8f0"
                strokeDasharray="4 5"
              />
              <text
                x={padding.left - 10}
                y={y + 4}
                textAnchor="end"
                fontSize="11"
                fill="#64748b"
              >
                {formatCompactSales(value)}
              </text>
            </g>
          );
        })}
        {items.map((item, index) => {
          const barHeight = (item.total_sales / maxValue) * chartHeight;
          const x = padding.left + index * slotWidth + (slotWidth - barWidth) / 2;
          const y = padding.top + chartHeight - barHeight;
          return (
            <g key={item.month}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                rx="2"
                fill="#0f766e"
                opacity={0.55 + 0.45 * (item.total_sales / maxValue)}
              >
                <title>
                  {formatMonth(item.month)}: {formatSales(item.total_sales)}
                </title>
              </rect>
              {index % labelStep === 0 || index === items.length - 1 ? (
                <text
                  x={x + barWidth / 2}
                  y={height - 22}
                  textAnchor="end"
                  transform={`rotate(-45 ${x + barWidth / 2} ${height - 22})`}
                  fontSize="10"
                  fill="#64748b"
                >
                  {formatMonth(item.month)}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function WeekdayChart({ items }: { items: WeekdayItem[] }) {
  if (!items.length) return null;
  const maxValue = Math.max(...items.map((item) => item.average_sales), 1);

  return (
    <div className="mt-6 space-y-5">
      {items.map((item) => {
        const width = Math.max(0, (item.average_sales / maxValue) * 100);
        return (
          <div key={item.day}>
            <div className="mb-2 flex items-center justify-between gap-3 text-sm">
              <span className="font-semibold text-slate-700">{item.day}</span>
              <span className="tabular-nums text-slate-600">
                {formatSales(item.average_sales)}
              </span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-teal-700 transition-all"
                style={{ width: `${width}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md">
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm font-medium text-slate-500">{label}</p>
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-teal-50 text-lg font-bold text-teal-700">
          {icon}
        </span>
      </div>
      <p className="mt-4 break-words text-2xl font-bold tracking-tight text-slate-950">
        {value}
      </p>
      <p className="mt-2 text-xs leading-5 text-slate-500">{detail}</p>
    </div>
  );
}

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="mb-5">
      <p className="text-sm font-semibold text-teal-700">{eyebrow}</p>
      <h3 className="mt-1 text-xl font-bold tracking-tight text-slate-950">
        {title}
      </h3>
      <p className="mt-1 text-sm leading-6 text-slate-500">{description}</p>
    </div>
  );
}

export default function Home() {
  const [days, setDays] = useState(7);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [performance, setPerformance] = useState<PerformanceResponse | null>(null);
  const [performanceError, setPerformanceError] = useState("");
  const [historical, setHistorical] = useState<HistoricalResponse | null>(null);
  const [historicalError, setHistoricalError] = useState("");
  const [historicalLoading, setHistoricalLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadDashboardData() {
      setHistoricalLoading(true);
      setHistoricalError("");
      setPerformanceError("");

      const [historicalResult, performanceResult] = await Promise.allSettled([
        fetch(`${API_BASE}/historical-analysis`),
        fetch(`${API_BASE}/model-performance`),
      ]);

      if (cancelled) return;

      if (historicalResult.status === "fulfilled") {
        try {
          const response = historicalResult.value;
          if (!response.ok) throw new Error(`Historical API returned ${response.status}`);
          setHistorical((await response.json()) as HistoricalResponse);
        } catch (err) {
          setHistoricalError(
            err instanceof Error ? err.message : "Could not load historical analysis.",
          );
        }
      } else {
        setHistoricalError("Could not connect to the historical analysis API.");
      }

      if (performanceResult.status === "fulfilled") {
        try {
          const response = performanceResult.value;
          if (!response.ok) throw new Error(`Model performance API returned ${response.status}`);
          setPerformance((await response.json()) as PerformanceResponse);
        } catch (err) {
          setPerformanceError(
            err instanceof Error ? err.message : "Could not load model performance.",
          );
        }
      } else {
        setPerformanceError("Could not connect to the model performance API.");
      }

      if (!cancelled) setHistoricalLoading(false);
    }

    void loadDashboardData();
    return () => {
      cancelled = true;
    };
  }, []);

  async function generateForecast() {
    setLoading(true);
    setError("");
    setForecast(null);

    try {
      const response = await fetch(`${API_BASE}/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ days }),
      });

      if (!response.ok) {
        let message = "Could not generate forecast. Check that the backend is running.";
        try {
          const details = await response.json();
          if (details?.detail) message = String(details.detail);
        } catch {
          // Keep the default error message if response is not JSON.
        }
        throw new Error(message);
      }

      setForecast((await response.json()) as ForecastResponse);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong. Please check the backend.",
      );
    } finally {
      setLoading(false);
    }
  }

  const items = forecast?.forecasts ?? [];
  const stats = useMemo(() => {
    if (!items.length) return null;
    const values = items.map((item) => Number(item.predicted_sales) || 0);
    const total = values.reduce((sum, value) => sum + value, 0);
    const average = total / values.length;
    const highestValue = Math.max(...values);
    const highestIndex = values.indexOf(highestValue);
    return {
      total,
      average,
      highestValue,
      highestDate: items[highestIndex]?.date ?? "—",
    };
  }, [items]);

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <header className="relative overflow-hidden bg-slate-950 text-white">
        <div className="absolute -right-20 -top-28 h-80 w-80 rounded-full bg-teal-500/10 blur-3xl" />
        <div className="absolute -bottom-32 left-1/3 h-64 w-64 rounded-full bg-cyan-400/10 blur-3xl" />
        <div className="relative mx-auto flex max-w-7xl flex-col gap-6 px-6 py-8 sm:px-8 lg:flex-row lg:items-center lg:justify-between lg:py-10">
          <div>
            <p className="text-xs font-bold tracking-[0.28em] text-teal-400 sm:text-sm">
              SALES INTELLIGENCE PLATFORM
            </p>
            <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
              SalesSense <span className="text-teal-400">AI</span>
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300 sm:text-base">
              AI-Powered Sales Forecasting &amp; Trend Intelligence System
            </p>
          </div>
          <div className="flex w-fit items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
            <span className="h-3 w-3 rounded-full bg-emerald-400" />
            <div>
              <p className="text-sm font-semibold">Forecasting Workspace</p>
              <p className="text-xs text-slate-400">XGBoost model selected</p>
            </div>
          </div>
        </div>
      </header>

      <section className="mx-auto max-w-7xl px-5 py-8 sm:px-8 sm:py-10">
        <div className="mb-7">
          <p className="text-sm font-semibold text-teal-700">OVERVIEW</p>
          <h2 className="mt-1 text-2xl font-bold tracking-tight sm:text-3xl">
            Sales Forecast Dashboard
          </h2>
          <p className="mt-2 text-sm text-slate-600 sm:text-base">
            Generate predictions and explore historical trends with model evaluation analytics.
          </p>
        </div>

        <div className="grid gap-5 lg:grid-cols-3">
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm lg:col-span-2">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-bold">Generate a Forecast</h3>
                <p className="mt-1 text-sm text-slate-500">
                  Select a forecast period to begin.
                </p>
              </div>
              <div className="rounded-xl bg-teal-50 p-3 text-teal-700">
                <span className="text-2xl">↗</span>
              </div>
            </div>
            <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-end">
              <div className="w-full sm:max-w-xs">
                <label
                  htmlFor="days"
                  className="mb-2 block text-sm font-semibold text-slate-700"
                >
                  Forecast Period
                </label>
                <select
                  id="days"
                  value={days}
                  onChange={(event) => setDays(Number(event.target.value))}
                  className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition focus:border-teal-600 focus:ring-4 focus:ring-teal-600/10"
                >
                  <option value={7}>Next 7 days</option>
                  <option value={14}>Next 14 days</option>
                  <option value={30}>Next 30 days</option>
                </select>
              </div>
              <button
                onClick={generateForecast}
                disabled={loading}
                className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-teal-700 px-6 py-3 text-sm font-bold text-white shadow-sm transition hover:bg-teal-800 focus:outline-none focus:ring-4 focus:ring-teal-600/20 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                    Generating...
                  </>
                ) : (
                  <>
                    Generate Forecast <span aria-hidden="true">→</span>
                  </>
                )}
              </button>
            </div>
            {error && (
              <div
                role="alert"
                className="mt-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
              >
                <p className="font-semibold">Forecast could not be generated</p>
                <p className="mt-1 break-words">{error}</p>
              </div>
            )}
            {loading && (
              <p className="mt-4 text-sm text-slate-500">
                The model is preparing your predictions. Please wait...
              </p>
            )}
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <p className="text-sm font-medium text-slate-500">Selected Model</p>
            <div className="mt-3 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-slate-950 text-sm font-extrabold text-teal-300">
                XGB
              </div>
              <div>
                <h3 className="text-xl font-bold">XGBoost</h3>
                <p className="mt-1 text-xs text-emerald-700">Backend model</p>
              </div>
            </div>
            <div className="mt-5 border-t border-slate-100 pt-4">
              <p className="text-sm leading-6 text-slate-600">
                Machine learning forecasts based on historical sales patterns and engineered time-series features.
              </p>
            </div>
          </section>
        </div>

        {stats && (
          <section className="mt-8" aria-label="Forecast summary">
            <SectionHeading
              eyebrow="KEY METRICS"
              title="Forecast Summary"
              description="Summary of the predictions generated for the selected period."
            />
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                label="Total Predicted Sales"
                value={formatSales(stats.total)}
                detail={`Across ${items.length} forecast days`}
                icon="Σ"
              />
              <MetricCard
                label="Average Daily Sales"
                value={formatSales(stats.average)}
                detail="Average prediction per day"
                icon="↗"
              />
              <MetricCard
                label="Highest Predicted Sales"
                value={formatSales(stats.highestValue)}
                detail="Highest daily prediction"
                icon="★"
              />
              <MetricCard
                label="Highest-Sales Date"
                value={stats.highestDate}
                detail="Date with the highest prediction"
                icon="▦"
              />
            </div>
          </section>
        )}

        {forecast && (
          <section className="mt-8 grid gap-6">
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-teal-700">FORECAST ANALYTICS</p>
                  <h3 className="mt-1 text-xl font-bold">Predicted Sales Trend</h3>
                  <p className="mt-1 text-sm text-slate-500">
                    Daily predicted sales from your XGBoost model.
                  </p>
                </div>
                <span className="rounded-full bg-teal-50 px-3 py-1.5 text-xs font-bold text-teal-800">
                  {forecast.forecast_days ?? items.length} days
                </span>
              </div>
              <SalesChart items={items} />
            </div>

            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-5 py-5 sm:px-7">
                <div>
                  <h3 className="text-xl font-bold">Forecast Results</h3>
                  <p className="mt-1 text-sm text-slate-500">
                    Model: {forecast.model ?? "XGBoost"} · Forecast period: {forecast.forecast_days ?? items.length} days
                  </p>
                </div>
                <span className="rounded-lg bg-slate-100 px-3 py-2 text-xs font-semibold text-slate-600">
                  {items.length} predictions
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[420px] text-left text-sm">
                  <thead className="bg-slate-50 text-xs uppercase tracking-wider text-slate-500">
                    <tr>
                      <th className="px-5 py-4 font-bold sm:px-7">Date</th>
                      <th className="px-5 py-4 text-right font-bold sm:px-7">Predicted Sales</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((item, index) => (
                      <tr
                        key={`${item.date}-${index}`}
                        className="border-t border-slate-100 transition hover:bg-slate-50"
                      >
                        <td className="px-5 py-4 font-medium text-slate-700 sm:px-7">
                          <span className="mr-3 inline-flex h-7 w-7 items-center justify-center rounded-lg bg-slate-100 text-xs font-bold text-slate-500">
                            {index + 1}
                          </span>
                          {item.date}
                        </td>
                        <td className="px-5 py-4 text-right font-bold tabular-nums text-slate-900 sm:px-7">
                          {formatSales(Number(item.predicted_sales) || 0)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="border-t border-slate-100 bg-slate-50 px-5 py-3 text-xs leading-5 text-slate-500 sm:px-7">
                These values are model predictions, not guaranteed actual sales. Dates are supplied by the backend.
              </div>
            </div>
          </section>
        )}

        {/* Historical Sales Analysis */}
        <section className="mt-12" aria-label="Historical sales analysis">
          <SectionHeading
            eyebrow="HISTORICAL DATA"
            title="Historical Sales Analysis"
            description={
              historical?.date_range
                ? `Actual daily sales aggregated from ${historical.date_range.start} to ${historical.date_range.end}.`
                : "Explore monthly sales trends and weekday patterns from the dataset."
            }
          />

          {historicalLoading && (
            <div className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-500 shadow-sm">
              Loading historical sales analysis...
            </div>
          )}

          {historicalError && !historicalLoading && (
            <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-800">
              <p className="font-semibold">Historical analysis could not be loaded</p>
              <p className="mt-1">{historicalError}</p>
              <p className="mt-2">Check that the FastAPI backend is running at {API_BASE}.</p>
            </div>
          )}

          {historical && !historicalLoading && (
            <>
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <MetricCard
                  label="Historical Total Sales"
                  value={formatCompactSales(historical.overall?.total_sales ?? 0)}
                  detail="Across all recorded dates"
                  icon="Σ"
                />
                <MetricCard
                  label="Average Daily Sales"
                  value={formatSales(historical.overall?.average_daily_sales ?? 0)}
                  detail="Average across historical days"
                  icon="↗"
                />
                <MetricCard
                  label="Peak Sales Day"
                  value={historical.overall?.highest_sales_day?.date ?? "—"}
                  detail={formatSales(historical.overall?.highest_sales_day?.sales ?? 0) + " sales"}
                  icon="★"
                />
                <MetricCard
                  label="Highest-Sales Month"
                  value={historical.insights?.highest_total_sales_month?.month
                    ? formatMonth(historical.insights.highest_total_sales_month.month)
                    : "—"}
                  detail={
                    "Total: " +
                    formatSales(historical.insights?.highest_total_sales_month?.total_sales ?? 0)
                  }
                  icon="▦"
                />
              </div>

              <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
                <SectionHeading
                  eyebrow="MONTHLY TREND"
                  title="Historical Monthly Sales"
                  description="Monthly total sales across the available dataset. Hover over a bar to inspect its value."
                />
                <MonthlyChart items={historical.monthly ?? []} />
              </div>

              <div className="mt-6 grid gap-6 lg:grid-cols-2">
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
                  <SectionHeading
                    eyebrow="WEEKLY PATTERN"
                    title="Average Sales by Weekday"
                    description="Compare average daily sales from Monday through Sunday."
                  />
                  <WeekdayChart items={historical.weekdays ?? []} />
                </div>

                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
                  <SectionHeading
                    eyebrow="DATA-DRIVEN INSIGHTS"
                    title="What the Data Shows"
                    description="Insights calculated by the backend from historical sales."
                  />
                  <div className="space-y-4">
                    <div className="rounded-xl bg-teal-50 p-4">
                      <p className="text-xs font-bold uppercase tracking-wide text-teal-800">
                        Highest monthly sales
                      </p>
                      <p className="mt-1 text-lg font-bold text-slate-900">
                        {historical.insights?.highest_total_sales_month?.month
                          ? formatMonth(historical.insights.highest_total_sales_month.month)
                          : "—"}
                      </p>
                      <p className="mt-1 text-sm text-slate-600">
                        Total sales: {formatSales(
                          historical.insights?.highest_total_sales_month?.total_sales ?? 0,
                        )}
                      </p>
                    </div>
                    <div className="rounded-xl bg-slate-50 p-4">
                      <p className="text-xs font-bold uppercase tracking-wide text-slate-500">
                        Lowest monthly sales
                      </p>
                      <p className="mt-1 text-lg font-bold text-slate-900">
                        {historical.insights?.lowest_total_sales_month?.month
                          ? formatMonth(historical.insights.lowest_total_sales_month.month)
                          : "—"}
                      </p>
                      <p className="mt-1 text-sm text-slate-600">
                        Total sales: {formatSales(
                          historical.insights?.lowest_total_sales_month?.total_sales ?? 0,
                        )}
                      </p>
                    </div>
                    <div className="rounded-xl border border-slate-200 p-4">
                      <p className="text-xs font-bold uppercase tracking-wide text-slate-500">
                        Strongest average weekday
                      </p>
                      <p className="mt-1 text-lg font-bold text-slate-900">
                        {historical.insights?.highest_average_sales_weekday?.day ?? "—"}
                      </p>
                      <p className="mt-1 text-sm text-slate-600">
                        Average sales: {formatSales(
                          historical.insights?.highest_average_sales_weekday?.average_sales ?? 0,
                        )}
                      </p>
                    </div>
                    <div className="rounded-xl border border-slate-200 p-4">
                      <p className="text-xs font-bold uppercase tracking-wide text-slate-500">
                        Lowest average weekday
                      </p>
                      <p className="mt-1 text-lg font-bold text-slate-900">
                        {historical.insights?.lowest_average_sales_weekday?.day ?? "—"}
                      </p>
                      <p className="mt-1 text-sm text-slate-600">
                        Average sales: {formatSales(
                          historical.insights?.lowest_average_sales_weekday?.average_sales ?? 0,
                        )}
                      </p>
                    </div>
                  </div>
                  <p className="mt-4 text-xs leading-5 text-slate-500">
                    {historical.note ?? "Values are aggregated from the historical dataset; they are not forecasts."}
                  </p>
                </div>
              </div>
            </>
          )}
        </section>

        {/* Model Performance Comparison */}
        <section className="mt-12" aria-label="Model performance comparison">
          <SectionHeading
            eyebrow="MODEL EVALUATION"
            title="Model Performance Comparison"
            description="Compare the saved evaluation results for the baseline model and XGBoost. Lower values indicate smaller forecast errors."
          />

          {performanceError && (
            <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-800">
              <p className="font-semibold">Model performance could not be loaded</p>
              <p className="mt-1">{performanceError}</p>
            </div>
          )}

          {performance && (
            <>
              <div className="grid gap-4 md:grid-cols-3">
                <MetricCard
                  label="MAE Improvement"
                  value={`${performance.improvements?.mae_improvement_percent ?? 0}%`}
                  detail="Lower mean absolute error with XGBoost"
                  icon="↘"
                />
                <MetricCard
                  label="RMSE Improvement"
                  value={`${performance.improvements?.rmse_improvement_percent ?? 0}%`}
                  detail="Lower root mean squared error with XGBoost"
                  icon="✓"
                />
                <MetricCard
                  label="WAPE Reduction"
                  value={`${performance.improvements?.wape_reduction_percentage_points ?? 0} pp`}
                  detail="Percentage-point reduction versus baseline"
                  icon="%"
                />
              </div>

              <div className="mt-6 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                <div className="border-b border-slate-100 px-5 py-5 sm:px-7">
                  <h4 className="text-lg font-bold">Evaluation Metrics</h4>
                  <p className="mt-1 text-sm text-slate-500">
                    Results from the saved model evaluation CSV.
                  </p>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[600px] text-left text-sm">
                    <thead className="bg-slate-50 text-xs uppercase tracking-wider text-slate-500">
                      <tr>
                        <th className="px-5 py-4 font-bold sm:px-7">Model</th>
                        <th className="px-5 py-4 text-right font-bold">MAE</th>
                        <th className="px-5 py-4 text-right font-bold">RMSE</th>
                        <th className="px-5 py-4 text-right font-bold sm:px-7">WAPE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {performance.models.map((item) => (
                        <tr key={item.model} className="border-t border-slate-100 hover:bg-slate-50">
                          <td className="px-5 py-4 font-semibold sm:px-7">{item.model}</td>
                          <td className="px-5 py-4 text-right tabular-nums">{formatSales(item.mae)}</td>
                          <td className="px-5 py-4 text-right tabular-nums">{formatSales(item.rmse)}</td>
                          <td className="px-5 py-4 text-right tabular-nums sm:px-7">{item.wape_percent.toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="border-t border-slate-100 bg-slate-50 px-5 py-3 text-xs leading-5 text-slate-500 sm:px-7">
                  {performance.note ?? "These are saved evaluation metrics and are not recalculated for each forecast request."}
                </div>
              </div>
            </>
          )}
        </section>

        {!forecast && (
          <section className="mt-8 rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-10 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-50 text-2xl text-teal-700">
              ↗
            </div>
            <h3 className="mt-4 font-bold">Your Forecast Will Appear Here</h3>
            <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-slate-500">
              Select a forecast period and click Generate Forecast to see the sales chart, summary metrics, and daily predictions.
            </p>
          </section>
        )}

        <footer className="mt-10 border-t border-slate-200 py-6 text-center text-sm text-slate-500">
          <p className="font-semibold text-slate-600">SalesSense AI</p>
          <p className="mt-1">
            Data-driven sales forecasting · Next.js · FastAPI · XGBoost
          </p>
        </footer>
      </section>
    </main>
  );
}
