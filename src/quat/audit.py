from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SUPPORTED_SUFFIXES = {".csv", ".txt", ".xlsx", ".xls", ".parquet"}

ROLE_ALIASES: dict[str, set[str]] = {
    "timestamp": {
        "timestamp", "datetime", "date time", "date_time", "period",
        "дата время", "дата_время", "время", "дата", "период",
    },
    "meter_id": {
        "meter id", "meter_id", "meter", "counter", "device id",
        "device_id", "object", "объект", "счетчик", "счётчик", "ячейка",
        "точка учета", "точка учёта", "присоединение",
    },
    "active_energy_kwh": {
        "active energy", "active_energy", "active energy kwh",
        "active_energy_kwh", "energy kwh", "energy_kwh", "consumption",
        "delta", "a+", "активная энергия", "активная_энергия", "расход",
        "потребление", "квтч", "квт ч",
    },
    "reactive_energy_kvarh": {
        "reactive energy", "reactive_energy", "reactive energy kvarh",
        "reactive_energy_kvarh", "q+", "реактивная энергия",
        "реактивная_энергия", "кварч", "квар ч",
    },
    "power_factor": {
        "power factor", "power_factor", "cos phi", "cos_phi", "cosφ",
        "cos ф", "cos_ф", "коэффициент мощности",
    },
}

DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
)


def _normalise(value: str) -> str:
    value = value.strip().lower().replace("ё", "е")
    value = re.sub(r"[\[\](){}]", " ", value)
    value = re.sub(r"[-_/\\]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


NORMALISED_ALIASES = {
    role: {_normalise(alias) for alias in aliases}
    for role, aliases in ROLE_ALIASES.items()
}


def detect_encoding(sample: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def infer_columns(headers: Sequence[str], configured: Mapping[str, Any]) -> dict[str, str | None]:
    exact = {_normalise(header): header for header in headers}
    result: dict[str, str | None] = {}
    for role in ROLE_ALIASES:
        requested = configured.get(role)
        if requested:
            result[role] = requested if requested in headers else None
            continue
        matches = [
            original
            for normalised, original in exact.items()
            if normalised in NORMALISED_ALIASES[role]
        ]
        result[role] = matches[0] if len(matches) == 1 else None
    return result


def parse_datetime(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        pass
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_number(value: str) -> float | None:
    text = value.strip().replace("\u00a0", "").replace(" ", "")
    if not text:
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


@dataclass
class NumericProfile:
    non_empty: int = 0
    parsed: int = 0
    invalid: int = 0
    zeros: int = 0
    negatives: int = 0
    minimum: float | None = None
    maximum: float | None = None
    total: float = 0.0

    def add(self, raw: str) -> None:
        if not raw.strip():
            return
        self.non_empty += 1
        number = parse_number(raw)
        if number is None:
            self.invalid += 1
            return
        self.parsed += 1
        self.zeros += number == 0
        self.negatives += number < 0
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)
        self.total += number

    def as_dict(self) -> dict[str, Any]:
        return {
            "non_empty": self.non_empty,
            "parsed": self.parsed,
            "invalid": self.invalid,
            "zeros": self.zeros,
            "negatives": self.negatives,
            "min": self.minimum,
            "max": self.maximum,
            "sum": self.total if self.parsed else None,
        }


def _iter_csv(path: Path, encoding: str, dialect: csv.Dialect) -> tuple[list[str], Iterator[dict[str, str]]]:
    handle = path.open("r", encoding=encoding, newline="")
    reader = csv.DictReader(handle, dialect=dialect)
    headers = reader.fieldnames or []

    def rows() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                yield {str(key): (value or "") for key, value in row.items() if key is not None}
        finally:
            handle.close()

    return headers, rows()


def profile_csv(path: Path, configured_columns: Mapping[str, Any]) -> dict[str, Any]:
    sample_bytes = path.read_bytes()[:65536]
    encoding = detect_encoding(sample_bytes)
    sample_text = sample_bytes.decode(encoding, errors="replace")
    dialect = detect_dialect(sample_text)
    headers, rows = _iter_csv(path, encoding, dialect)
    roles = infer_columns(headers, configured_columns)

    empty = Counter()
    row_count = 0
    malformed_rows = 0
    meter_values: set[str] = set()
    meter_overflow = False
    last_timestamp: dict[str, datetime] = {}
    interval_seconds: Counter[int] = Counter()
    timestamp_parsed = 0
    timestamp_invalid = 0
    timestamp_duplicates = 0
    timestamp_out_of_order = 0
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None
    numeric = {
        role: NumericProfile()
        for role in ("active_energy_kwh", "reactive_energy_kvarh", "power_factor")
        if roles.get(role)
    }

    for row in rows:
        row_count += 1
        if len(row) != len(headers):
            malformed_rows += 1
        for header in headers:
            if not row.get(header, "").strip():
                empty[header] += 1

        meter_column = roles.get("meter_id")
        meter = row.get(meter_column, "").strip() if meter_column else "__single_meter__"
        if meter and len(meter_values) < 10000:
            meter_values.add(meter)
        elif meter and meter not in meter_values:
            meter_overflow = True

        timestamp_column = roles.get("timestamp")
        if timestamp_column:
            raw_timestamp = row.get(timestamp_column, "")
            parsed_timestamp = parse_datetime(raw_timestamp)
            if raw_timestamp.strip() and parsed_timestamp is None:
                timestamp_invalid += 1
            elif parsed_timestamp is not None:
                timestamp_parsed += 1
                timestamp_min = parsed_timestamp if timestamp_min is None else min(timestamp_min, parsed_timestamp)
                timestamp_max = parsed_timestamp if timestamp_max is None else max(timestamp_max, parsed_timestamp)
                previous = last_timestamp.get(meter)
                if previous is not None:
                    delta = int((parsed_timestamp - previous).total_seconds())
                    if delta == 0:
                        timestamp_duplicates += 1
                    elif delta < 0:
                        timestamp_out_of_order += 1
                    else:
                        interval_seconds[delta] += 1
                last_timestamp[meter] = parsed_timestamp

        for role, profile in numeric.items():
            column = roles[role]
            if column:
                profile.add(row.get(column, ""))

    top_intervals = [
        {"seconds": seconds, "minutes": seconds / 60, "count": count}
        for seconds, count in interval_seconds.most_common(10)
    ]
    warnings: list[str] = []
    for required in ("timestamp", "meter_id", "active_energy_kwh"):
        if roles.get(required) is None:
            warnings.append(f"Не определён столбец роли '{required}'. Укажите его в конфигурации.")
    if timestamp_invalid:
        warnings.append(f"Не распознано временных меток: {timestamp_invalid}.")
    if timestamp_out_of_order:
        warnings.append("Строки не полностью упорядочены по времени внутри счётчика.")
    active = numeric.get("active_energy_kwh")
    if active and active.negatives:
        warnings.append("Обнаружены отрицательные значения активной энергии; проверьте сбросы, направление и тип показания.")

    return {
        "path": str(path),
        "format": "csv",
        "size_bytes": path.stat().st_size,
        "encoding": encoding,
        "delimiter": getattr(dialect, "delimiter", ","),
        "rows": row_count,
        "columns": headers,
        "roles": roles,
        "empty_cells_by_column": dict(empty),
        "malformed_rows": malformed_rows,
        "meter_count": len(meter_values),
        "meter_count_capped": meter_overflow,
        "meter_examples": sorted(meter_values)[:20],
        "timestamps": {
            "parsed": timestamp_parsed,
            "invalid": timestamp_invalid,
            "min": timestamp_min.isoformat() if timestamp_min else None,
            "max": timestamp_max.isoformat() if timestamp_max else None,
            "adjacent_duplicates": timestamp_duplicates,
            "out_of_order": timestamp_out_of_order,
            "top_intervals": top_intervals,
        },
        "numeric": {role: profile.as_dict() for role, profile in numeric.items()},
        "warnings": warnings,
    }


def profile_optional_format(path: Path) -> dict[str, Any]:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return {
            "path": str(path),
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
            "status": "dependency_missing",
            "warnings": [
                "Для анализа этого формата установите зависимости: python -m pip install -e .[data]"
            ],
        }

    if path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        return {
            "path": str(path),
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
            "status": "inventory_only",
            "sheets": workbook.sheet_names,
            "warnings": ["В первом проходе выполнена инвентаризация листов; после выбора листа нужен явный маппинг столбцов."],
        }
    frame = pd.read_parquet(path)
    return {
        "path": str(path),
        "format": "parquet",
        "size_bytes": path.stat().st_size,
        "status": "inventory_only",
        "rows": len(frame),
        "columns": [str(column) for column in frame.columns],
        "warnings": ["Для полного профиля Parquet после инвентаризации нужен явный маппинг столбцов."],
    }


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"timezone": "Asia/Qyzylorda", "value_kind": "unknown", "columns": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_directory(input_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    files = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    profiles: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() in {".csv", ".txt"}:
            profiles.append(profile_csv(path, config.get("columns", {})))
        else:
            profiles.append(profile_optional_format(path))
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_directory": str(input_dir),
        "timezone": config.get("timezone"),
        "declared_value_kind": config.get("value_kind"),
        "file_count": len(files),
        "files": profiles,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Аудит данных Quat",
        "",
        f"Файлов найдено: **{report['file_count']}**",
        f"Заявленная часовая зона: **{report.get('timezone') or 'не указана'}**",
        f"Тип значения: **{report.get('declared_value_kind') or 'не указан'}**",
        "",
    ]
    if not report["files"]:
        lines.extend([
            "В `data/raw` пока нет поддерживаемых файлов.",
            "",
            "Поддерживаются CSV, TXT, XLSX, XLS и Parquet.",
        ])
        return "\n".join(lines) + "\n"

    for profile in report["files"]:
        lines.extend([
            f"## {Path(profile['path']).name}",
            "",
            f"- Формат: `{profile['format']}`",
            f"- Размер: `{profile['size_bytes']}` байт",
        ])
        if "rows" in profile:
            lines.append(f"- Строк: `{profile['rows']}`")
        timestamps = profile.get("timestamps")
        if timestamps:
            lines.extend([
                f"- Период: `{timestamps.get('min')}` — `{timestamps.get('max')}`",
                f"- Распознано временных меток: `{timestamps.get('parsed')}`",
                f"- Счётчиков/объектов: `{profile.get('meter_count')}`",
            ])
            intervals = timestamps.get("top_intervals") or []
            if intervals:
                lines.append(f"- Наиболее частый шаг: `{intervals[0]['minutes']:g}` минут")
        roles = profile.get("roles")
        if roles:
            lines.extend(["", "Распознанные роли столбцов:", ""])
            for role, column in roles.items():
                lines.append(f"- `{role}` → `{column}`")
        warnings = profile.get("warnings") or []
        if warnings:
            lines.extend(["", "Предупреждения:", ""])
            lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

