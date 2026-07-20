from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quat.tariffs import (
    apply_tariffs,
    price_candidate_events,
    schedule_from_payload,
    validate_contiguous_coverage,
)


def payload() -> dict[str, object]:
    return {
        "currency": "KZT",
        "selected_consumer_group": 2,
        "periods": [
            {
                "start": "2025-01-01",
                "end_exclusive": "2025-02-01",
                "group_2": {
                    "rate_with_vat_kzt_per_kwh": 10.0,
                    "status": "published",
                },
                "group_3": {
                    "rate_with_vat_kzt_per_kwh": 20.0,
                    "status": "published",
                },
            },
            {
                "start": "2025-02-01",
                "end_exclusive": "2025-03-01",
                "group_2": {
                    "rate_with_vat_kzt_per_kwh": 12.0,
                    "status": "derived",
                },
                "group_3": {
                    "rate_with_vat_kzt_per_kwh": 24.0,
                    "status": "derived",
                },
            },
        ],
    }


class TariffTests(unittest.TestCase):
    def test_period_boundary_and_cost(self) -> None:
        schedule = schedule_from_payload(payload())
        frame = pd.DataFrame({
            "timestamp_local": ["2025-01-31 23:00", "2025-02-01 00:00"],
            "active_import_kwh": [2.0, 3.0],
        })
        priced = apply_tariffs(frame, schedule)
        self.assertEqual(priced["tariff_rate_kzt_per_kwh"].tolist(), [10.0, 12.0])
        self.assertEqual(priced["energy_cost_kzt"].tolist(), [20.0, 36.0])

    def test_group_override(self) -> None:
        schedule = schedule_from_payload(payload(), consumer_group=3)
        self.assertEqual(schedule.rate_at("2025-01-15").rate_with_vat_kzt_per_kwh, 20.0)

    def test_uncovered_timestamp_raises(self) -> None:
        schedule = schedule_from_payload(payload())
        frame = pd.DataFrame({
            "timestamp_local": ["2025-03-01 00:00"],
            "active_import_kwh": [1.0],
        })
        with self.assertRaises(ValueError):
            apply_tariffs(frame, schedule)

    def test_candidate_event_pricing(self) -> None:
        schedule = schedule_from_payload(payload())
        events = pd.DataFrame({
            "start": [pd.Timestamp("2025-02-05 08:00")],
            "excess_kwh": [100.0],
        })
        priced = price_candidate_events(events, schedule)
        self.assertEqual(priced.loc[0, "estimated_excess_cost_kzt"], 1200.0)

    def test_contiguous_schedule_has_no_gaps(self) -> None:
        schedule = schedule_from_payload(payload())
        self.assertEqual(validate_contiguous_coverage(schedule.periods), [])


if __name__ == "__main__":
    unittest.main()
