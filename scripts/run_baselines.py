from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quat.baselines import benchmark_meter, candidate_events  # noqa: E402
from quat.tariffs import load_tariff_schedule, price_candidate_events  # noqa: E402


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    prepared = frame.copy()
    for column in prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared[column]):
            prepared[column] = prepared[column].map(
                lambda value: value.isoformat() if pd.notna(value) else None
            )
    prepared = prepared.replace({np.nan: None})
    return prepared.to_dict(orient="records")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark causal weekly energy baselines")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "hourly_energy.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
    )
    parser.add_argument(
        "--meter-registry",
        type=Path,
        default=PROJECT_ROOT / "config" / "meter_registry.csv",
    )
    parser.add_argument(
        "--tariffs",
        type=Path,
        default=PROJECT_ROOT / "config" / "electricity_tariffs.json",
    )
    parser.add_argument("--consumer-group", type=int, choices=[2, 3])
    parser.add_argument("--validation-weeks", type=int, default=8)
    parser.add_argument(
        "--weather",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "turkestan_weather.json",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, low_memory=False)
    data["timestamp_local"] = pd.to_datetime(data["timestamp_local"])
    registry = pd.read_csv(args.meter_registry)
    configured_names = set(
        registry.loc[_truthy(registry["include_in_load_model"]), "meter_name"]
    )
    available_names = set(data["meter_name"].dropna().unique())
    missing_configured = sorted(configured_names - available_names)
    tariff_schedule = load_tariff_schedule(
        args.tariffs,
        consumer_group=args.consumer_group,
    )
    weather_payload = json.loads(args.weather.read_text(encoding="utf-8"))
    weather = pd.DataFrame({
        "timestamp_local": pd.to_datetime(weather_payload["hourly"]["time"]),
        "temperature_2m": weather_payload["hourly"]["temperature_2m"],
        "relative_humidity_2m": weather_payload["hourly"]["relative_humidity_2m"],
        "precipitation": weather_payload["hourly"]["precipitation"],
        "cloud_cover": weather_payload["hourly"]["cloud_cover"],
    })
    eligible = data[data["meter_name"].isin(configured_names)].copy()
    if eligible.empty:
        raise ValueError("No registry-enabled meters were found in the normalized data")
    end = eligible["timestamp_local"].max() + pd.Timedelta(hours=1)
    validation_start = end - pd.Timedelta(weeks=args.validation_weeks)

    metric_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    baseline_columns = {
        "last_week": "baseline_last_week",
        "median_previous_4_weeks": "baseline_median_4w",
        "weather_hour_of_week_ridge": "baseline_weather_hour_of_week",
    }
    for meter_name, meter_frame in eligible.groupby("meter_name", sort=True):
        modeled, metrics = benchmark_meter(meter_frame, validation_start, weather)
        for metric in metrics:
            metric_rows.append({
                "meter_name": meter_name,
                "validation_start": validation_start.isoformat(),
                "validation_end_exclusive": end.isoformat(),
                **metric,
            })
        best_metric = min(
            (metric for metric in metrics if pd.notna(metric["wmape"])),
            key=lambda metric: float(metric["wmape"]),
        )
        best_column = baseline_columns[str(best_metric["model"])]
        nonzero_share = float((meter_frame["active_import_kwh"] > 0).mean())
        events = (
            candidate_events(
                modeled,
                validation_start,
                baseline_column=best_column,
            )
            if nonzero_share >= 0.5
            else pd.DataFrame()
        )
        if not events.empty:
            model_wmape = float(best_metric["wmape"])
            events["model_validation_wmape"] = model_wmape
            events["model_quality_band"] = (
                "good" if model_wmape <= 0.15
                else "usable" if model_wmape <= 0.25
                else "weak"
            )
            events.insert(0, "meter_name", meter_name)
            event_frames.append(price_candidate_events(events, tariff_schedule))

    metrics_frame = pd.DataFrame(metric_rows)
    events_frame = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame()
    )
    if not events_frame.empty:
        events_frame = events_frame.sort_values("excess_kwh", ascending=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "baseline_metrics.csv"
    events_path = args.output_dir / "candidate_events.csv"
    summary_path = args.output_dir / "baseline_summary.json"
    metrics_frame.to_csv(metrics_path, index=False)
    events_frame.to_csv(events_path, index=False)

    best = (
        metrics_frame.sort_values(["meter_name", "wmape"])
        .groupby("meter_name", as_index=False)
        .first()
    )
    summary = {
        "validation_start": validation_start.isoformat(),
        "validation_end_exclusive": end.isoformat(),
        "validation_weeks": args.validation_weeks,
        "eligible_meter_count": int(eligible["meter_name"].nunique()),
        "configured_meter_count": len(configured_names),
        "missing_configured_meters": missing_configured,
        "consumer_group": tariff_schedule.consumer_group,
        "candidate_event_count": int(len(events_frame)),
        "candidate_excess_kwh": (
            float(events_frame["excess_kwh"].sum())
            if not events_frame.empty else 0.0
        ),
        "estimated_candidate_excess_cost_kzt": (
            float(events_frame["estimated_excess_cost_kzt"].sum())
            if not events_frame.empty else 0.0
        ),
        "best_models": best.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    model_analysis_path = args.output_dir / "model_analysis.json"
    model_analysis_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "metrics": _records(metrics_frame),
                "events": _records(events_frame),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Eligible meters: {summary['eligible_meter_count']}")
    print(f"Validation: {summary['validation_start']} — {summary['validation_end_exclusive']}")
    print(f"Candidate events: {summary['candidate_event_count']}")
    print(f"Candidate excess: {summary['candidate_excess_kwh']:,.0f} kWh")
    print(
        "Estimated candidate value: "
        f"{summary['estimated_candidate_excess_cost_kzt']:,.0f} KZT"
    )
    print(best[["meter_name", "model", "wmape", "mae_kwh", "bias"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
