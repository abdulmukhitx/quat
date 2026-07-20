from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quat.askue_import import AskueImportError, import_askue_folder  # noqa: E402


def _run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *arguments]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import the ASKUE Excel folder and refresh all Quat analytics"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.home() / "Desktop" / "askua",
        help="Folder containing one ASKUE Excel export per meter",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "hourly_energy.csv",
    )
    parser.add_argument(
        "--meter-registry",
        type=Path,
        default=PROJECT_ROOT / "config" / "meter_registry.csv",
    )
    parser.add_argument(
        "--import-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "askue_import.json",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit a subset of registry meters (not recommended for the dashboard)",
    )
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="Stop after writing the validated normalized CSV",
    )
    args = parser.parse_args()

    try:
        report = import_askue_folder(
            args.input_dir,
            args.output,
            args.meter_registry,
            args.import_report,
            allow_partial=args.allow_partial,
        )
    except AskueImportError as exc:
        print(f"ASKUE import failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"Imported {report['source_file_count']} files / "
        f"{report['row_count']:,} hourly rows / "
        f"{report['active_import_kwh']:,.0f} kWh"
    )
    if args.import_only:
        return 0

    _run("run_baselines.py", "--input", str(args.output))
    _run("build_portfolio_report.py", "--input", str(args.output))
    print("Quat analytics and dashboard data refreshed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
