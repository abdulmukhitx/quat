from __future__ import annotations

import unittest

import pandas as pd

from quat.askue_import import AskueImportError, parse_workbook_frame


def workbook_frame(*, summary_active: float = 30.0) -> pd.DataFrame:
    rows = [[None] * 9 for _ in range(21)]
    rows[3][1:4] = ["Начало", "2025-07-01", "00:00:00"]
    rows[3][6:9] = ["Конеч.дата", "2025-07-01", "02:00:00"]
    rows[6][1:9] = [
        "Клиент:",
        "University",
        "Объект:",
        "КРУН 10кВ",
        "Место:",
        "Ячейка №1А (Тест)",
        "Устройство:",
        "Satec EM133",
    ]
    rows[9][1:3] = ["Энергия", "Суммарное значение"]
    rows[10][1:3] = ["кВт*ч(импорт)", summary_active]
    rows[11][1:3] = ["квар*ч(импорт)", 7.0]
    rows[12][1:3] = ["кВт*ч(экспорт)", 1.0]
    rows[13][1:3] = ["квар*ч(экспорт)", 0.0]
    rows[17][1:9] = [
        "Дата",
        "С",
        "До",
        "кВт*ч(импорт)",
        "квар*ч(импорт)",
        "кВт*ч(экспорт)",
        "квар*ч(экспорт)",
        "cos ф",
    ]
    rows[18][1:9] = ["2025-07-01", "00:00", "01:00", 10, 3, 0, 0, 0.95]
    rows[19][1:9] = ["2025-07-01", "01:00", "02:00", 20, 4, 1, 0, 0.96]
    rows[20][1:9] = ["2025-07-01", "02:00", None, None, None, None, None, 0.96]
    return pd.DataFrame(rows)


class AskueImportTests(unittest.TestCase):
    def test_parse_workbook_frame_normalizes_hourly_rows(self) -> None:
        normalized, metadata = parse_workbook_frame(
            workbook_frame(),
            source_file="meter.xlsx",
            registry_ids={"Ячейка №1А (Тест)": "1A"},
            role_by_name={"Ячейка №1А (Тест)": "outgoing_load"},
        )

        self.assertEqual(metadata.meter_id, "1A")
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized["active_import_kwh"].sum(), 30)
        self.assertEqual(normalized["timestamp_local"].tolist(), [
            pd.Timestamp("2025-07-01T00:00:00"),
            pd.Timestamp("2025-07-01T01:00:00"),
        ])
        self.assertEqual(normalized["source_file"].unique().tolist(), ["meter.xlsx"])
        self.assertTrue(normalized["cos_phi_calculated"].between(0, 1).all())

    def test_parse_workbook_frame_rejects_summary_mismatch(self) -> None:
        with self.assertRaisesRegex(AskueImportError, "Summary mismatch"):
            parse_workbook_frame(
                workbook_frame(summary_active=31),
                source_file="meter.xlsx",
            )


if __name__ == "__main__":
    unittest.main()
