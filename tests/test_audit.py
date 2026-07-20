from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quat.audit import audit_directory, parse_number, profile_csv  # noqa: E402


class AuditTests(unittest.TestCase):
    def test_parse_number_supports_decimal_comma(self) -> None:
        self.assertEqual(parse_number("1 234,5"), 1234.5)
        self.assertEqual(parse_number("1,234.5"), 1234.5)
        self.assertIsNone(parse_number("нет данных"))

    def test_profiles_multiple_meters_and_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "energy.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["Дата_время", "Счётчик", "Активная энергия"])
                writer.writerow(["2026-01-01 00:00", "A", "10,0"])
                writer.writerow(["2026-01-01 00:15", "A", "11,0"])
                writer.writerow(["2026-01-01 00:00", "B", "4,0"])
                writer.writerow(["2026-01-01 00:15", "B", "-1,0"])

            profile = profile_csv(path, {})

            self.assertEqual(profile["rows"], 4)
            self.assertEqual(profile["meter_count"], 2)
            self.assertEqual(profile["timestamps"]["top_intervals"][0]["minutes"], 15)
            self.assertEqual(profile["numeric"]["active_energy_kwh"]["negatives"], 1)

    def test_empty_directory_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = audit_directory(Path(temporary), {"columns": {}})
            self.assertEqual(report["file_count"], 0)
            self.assertEqual(report["files"], [])


if __name__ == "__main__":
    unittest.main()

