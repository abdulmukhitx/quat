from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quat.portfolio import build_portfolio_analysis  # noqa: E402
from quat.tariffs import load_tariff_schedule  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build annual Quat portfolio analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "hourly_energy.csv",
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
    parser.add_argument(
        "--model-analysis",
        type=Path,
        default=PROJECT_ROOT / "reports" / "model_analysis.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "portfolio_analysis.json",
    )
    parser.add_argument(
        "--dashboard-data-dir",
        type=Path,
        default=PROJECT_ROOT / "dashboard" / "app" / "data",
        help="Write the latest analysis JSON into the Quat dashboard when present",
    )
    parser.add_argument("--consumer-group", type=int, choices=[2, 3])
    args = parser.parse_args()

    data = pd.read_csv(args.input, low_memory=False)
    registry = pd.read_csv(args.meter_registry)
    schedule = load_tariff_schedule(
        args.tariffs,
        consumer_group=args.consumer_group,
    )
    model_analysis = (
        json.loads(args.model_analysis.read_text(encoding="utf-8"))
        if args.model_analysis.exists()
        else None
    )
    analysis = build_portfolio_analysis(
        data,
        registry,
        schedule,
        model_analysis=model_analysis,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.dashboard_data_dir:
        args.dashboard_data_dir.mkdir(parents=True, exist_ok=True)
        (args.dashboard_data_dir / "portfolio_analysis.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if model_analysis is not None:
            (args.dashboard_data_dir / "model_analysis.json").write_text(
                json.dumps(model_analysis, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    summary = analysis["summary"]
    print(f"Master input energy: {summary['master_input_energy_kwh']:,.0f} kWh")
    print(f"Estimated gross cost: {summary['estimated_gross_cost_kzt']:,.0f} KZT")
    print(f"Modeled loads: {summary['modeled_load_meter_count']}")
    print(f"Candidate events: {summary['candidate_event_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
