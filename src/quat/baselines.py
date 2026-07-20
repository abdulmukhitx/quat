from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


HOURS_PER_WEEK = 24 * 7


def add_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal weekly baselines without using future observations."""
    result = frame.sort_values("timestamp_local").copy()
    actual = result["active_import_kwh"].astype(float)
    result["baseline_last_week"] = actual.shift(HOURS_PER_WEEK)
    weekly_history = pd.concat(
        [actual.shift(HOURS_PER_WEEK * weeks) for weeks in range(1, 5)],
        axis=1,
    )
    result["baseline_median_4w"] = weekly_history.median(axis=1, skipna=False)
    return result


def metric_row(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    valid = actual.notna() & predicted.notna()
    observed = actual.loc[valid].astype(float)
    forecast = predicted.loc[valid].astype(float)
    error = forecast - observed
    denominator = float(observed.abs().sum())
    return {
        "observations": int(valid.sum()),
        "mae_kwh": float(error.abs().mean()),
        "rmse_kwh": float(np.sqrt(np.mean(np.square(error)))),
        "wmape": float(error.abs().sum() / denominator) if denominator else np.nan,
        "bias": float(error.sum() / denominator) if denominator else np.nan,
    }


def benchmark_meter(
    frame: pd.DataFrame,
    validation_start: pd.Timestamp,
    weather: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, float | int | str]]]:
    modeled = add_baselines(frame)
    if weather is not None:
        modeled = modeled.merge(weather, on="timestamp_local", how="left", validate="one_to_one")
        training_mask = modeled["timestamp_local"] < validation_start
        modeled["baseline_weather_hour_of_week"] = fit_weather_hour_of_week(
            modeled.loc[training_mask],
            modeled,
        )
    validation = modeled[modeled["timestamp_local"] >= validation_start]
    metrics: list[dict[str, float | int | str]] = []
    models = [
        ("last_week", "baseline_last_week"),
        ("median_previous_4_weeks", "baseline_median_4w"),
    ]
    if weather is not None:
        models.append(("weather_hour_of_week_ridge", "baseline_weather_hour_of_week"))
    for model_name, column in models:
        metrics.append({
            "model": model_name,
            **metric_row(validation["active_import_kwh"], validation[column]),
        })
    return modeled, metrics


def _weather_matrix(frame: pd.DataFrame) -> np.ndarray:
    timestamp = frame["timestamp_local"]
    hour_of_week = timestamp.dt.dayofweek.to_numpy() * 24 + timestamp.dt.hour.to_numpy()
    hour_of_day = timestamp.dt.hour.to_numpy()
    temperature = frame["temperature_2m"].astype(float).to_numpy()
    weekly = np.eye(HOURS_PER_WEEK, dtype=float)[hour_of_week]
    hourly = np.eye(24, dtype=float)[hour_of_day]
    cooling = np.maximum(temperature - 20.0, 0.0)[:, None] * hourly
    heating = np.maximum(12.0 - temperature, 0.0)[:, None] * hourly
    return np.concatenate([weekly, cooling, heating], axis=1)


def fit_weather_hour_of_week(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Fit an interpretable hour-of-week baseline with hourly weather response."""
    required = training["active_import_kwh"].notna() & training["temperature_2m"].notna()
    train = training.loc[required]
    x_train = _weather_matrix(train)
    y_train = train["active_import_kwh"].astype(float).to_numpy()
    penalty = np.eye(x_train.shape[1], dtype=float) * ridge_alpha
    coefficients = np.linalg.solve(
        x_train.T @ x_train + penalty,
        x_train.T @ y_train,
    )
    x_prediction = _weather_matrix(prediction)
    return np.maximum(x_prediction @ coefficients, 0.0)


def candidate_events(
    modeled: pd.DataFrame,
    validation_start: pd.Timestamp,
    baseline_column: str = "baseline_median_4w",
    minimum_relative_excess: float = 0.25,
    quantile: float = 0.99,
) -> pd.DataFrame:
    """Return persistent positive residuals for human review, not confirmed faults."""
    result = modeled.copy()
    result["residual_kwh"] = (
        result["active_import_kwh"] - result[baseline_column]
    )
    training_residual = result.loc[
        (result["timestamp_local"] < validation_start)
        & result["residual_kwh"].notna(),
        "residual_kwh",
    ]
    positive = training_residual[training_residual > 0]
    if positive.empty:
        return pd.DataFrame()
    absolute_threshold = float(positive.quantile(quantile))
    validation = result[result["timestamp_local"] >= validation_start].copy()
    baseline_floor = max(
        float(result["active_import_kwh"].median()) * 0.05,
        1.0,
    )
    validation["relative_excess"] = validation["residual_kwh"] / validation[
        baseline_column
    ].clip(lower=baseline_floor)
    validation["candidate"] = (
        (validation["residual_kwh"] > absolute_threshold)
        & (validation["relative_excess"] >= minimum_relative_excess)
    )
    validation["candidate_run"] = (
        validation["candidate"] != validation["candidate"].shift(fill_value=False)
    ).cumsum()

    events: list[dict[str, object]] = []
    for _, group in validation[validation["candidate"]].groupby("candidate_run"):
        if len(group) < 2:
            continue
        events.append({
            "start": group["timestamp_local"].min(),
            "end": group["timestamp_local"].max() + pd.Timedelta(hours=1),
            "duration_hours": int(len(group)),
            "observed_kwh": float(group["active_import_kwh"].sum()),
            "baseline_kwh": float(group[baseline_column].sum()),
            "excess_kwh": float(group["residual_kwh"].sum()),
            "peak_hourly_excess_kwh": float(group["residual_kwh"].max()),
            "max_relative_excess": float(group["relative_excess"].max()),
            "threshold_kwh": absolute_threshold,
            "baseline_model": baseline_column.removeprefix("baseline_"),
        })
    return pd.DataFrame(events)
