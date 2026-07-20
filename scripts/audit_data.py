from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quat.audit import audit_directory, load_config, write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only аудит выгрузок АСКУЭ")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "data_contract.json")
    parser.add_argument("--json", type=Path, default=PROJECT_ROOT / "reports" / "data_audit.json")
    parser.add_argument("--markdown", type=Path, default=PROJECT_ROOT / "reports" / "data_audit.md")
    args = parser.parse_args()

    report = audit_directory(args.input, load_config(args.config))
    write_report(report, args.json, args.markdown)
    print(f"Файлов проверено: {report['file_count']}")
    print(f"JSON: {args.json}")
    print(f"Markdown: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

