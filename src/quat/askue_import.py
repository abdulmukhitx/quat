from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "timestamp_local",
    "meter_id",
    "meter_name",
    "object",
    "device",
    "meter_class",
    "active_import_kwh",
    "reactive_import_kvarh",
    "active_export_kwh",
    "reactive_export_kvarh",
    "cos_phi_calculated",
    "source_rows",
    "quality_flags",
    "source_file",
]


class AskueImportError(ValueError):
    """Raised when an ASKUE workbook cannot be normalized safely."""


@dataclass(frozen=True)
class WorkbookMetadata:
    meter_id: str
    meter_name: str
    object_name: str
    device: str
    period_start: pd.Timestamp
    period_end_exclusive: pd.Timestamp
    summary_totals: tuple[float, float, float, float]


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _find_label(frame: pd.DataFrame, label: str, limit: int = 18) -> tuple[int, int]:
    expected = label.casefold().rstrip(":").strip()
    for row_index in range(min(limit, len(frame))):
        for column_index, value in enumerate(frame.iloc[row_index]):
            candidate = _clean_text(value).casefold().rstrip(":").strip()
            if candidate == expected:
                return row_index, column_index
    raise AskueImportError(f"Required label '{label}' was not found")


def _value_after(frame: pd.DataFrame, label: str) -> object:
    row_index, column_index = _find_label(frame, label)
    if column_index + 1 >= frame.shape[1]:
        raise AskueImportError(f"No value follows label '{label}'")
    return frame.iat[row_index, column_index + 1]


def _parse_date_and_time(date_value: object, time_value: object) -> pd.Timestamp:
    parsed_date = _parse_date_value(date_value)
    if pd.isna(parsed_date):
        raise AskueImportError(f"Invalid report date: {date_value!r}")
    return pd.Timestamp(parsed_date).normalize() + _time_delta(time_value)


def _parse_date_value(value: object) -> pd.Timestamp | pd.NaT:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)
    text = _clean_text(value)
    if not text:
        return pd.NaT
    if re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?", text):
        return pd.to_datetime(text, errors="coerce", yearfirst=True)
    return pd.to_datetime(text, errors="coerce", dayfirst=True)


def _time_delta(value: object) -> pd.Timedelta:
    if isinstance(value, pd.Timestamp):
        value = value.time()
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return pd.Timedelta(
            hours=value.hour,
            minutes=value.minute,
            seconds=value.second,
        )
    if isinstance(value, (int, float, np.number)) and not pd.isna(value):
        return pd.Timedelta(days=float(value))
    text = _clean_text(value)
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if not match:
        raise AskueImportError(f"Invalid report time: {value!r}")
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if hours > 23 or minutes > 59 or seconds > 59:
        raise AskueImportError(f"Invalid report time: {value!r}")
    return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _extract_meter_id(meter_name: str) -> str:
    match = re.search(r"№\s*([0-9]+\s*[A-Za-zА-Яа-я]?)", meter_name)
    if not match:
        raise AskueImportError(f"Cannot extract meter id from '{meter_name}'")
    meter_id = re.sub(r"\s+", "", match.group(1)).upper()
    return meter_id.translate(str.maketrans({"А": "A", "В": "B", "С": "C"}))


def _numeric_total(frame: pd.DataFrame, row: int, column: int) -> float:
    value = pd.to_numeric(pd.Series([frame.iat[row, column]]), errors="coerce").iloc[0]
    if pd.isna(value):
        raise AskueImportError(f"Invalid summary total at row {row + 1}")
    return float(value)


def extract_metadata(
    frame: pd.DataFrame,
    registry_ids: dict[str, str] | None = None,
) -> WorkbookMetadata:
    meter_name = _clean_text(_value_after(frame, "Место"))
    object_name = _clean_text(_value_after(frame, "Объект"))
    device = _clean_text(_value_after(frame, "Устройство"))
    if not meter_name:
        raise AskueImportError("Meter name is empty")

    start_row, start_column = _find_label(frame, "Начало")
    end_row, end_column = _find_label(frame, "Конеч.дата")
    period_start = _parse_date_and_time(
        frame.iat[start_row, start_column + 1],
        frame.iat[start_row, start_column + 2],
    )
    period_end_exclusive = _parse_date_and_time(
        frame.iat[end_row, end_column + 1],
        frame.iat[end_row, end_column + 2],
    )
    if period_end_exclusive <= period_start:
        raise AskueImportError("Report end must be later than report start")

    energy_row, energy_column = _find_label(frame, "Энергия")
    if energy_column + 1 >= frame.shape[1] or energy_row + 4 >= len(frame):
        raise AskueImportError("Energy summary block is incomplete")
    summary_totals = tuple(
        _numeric_total(frame, energy_row + offset, energy_column + 1)
        for offset in range(1, 5)
    )
    meter_id = (
        registry_ids[meter_name]
        if registry_ids and meter_name in registry_ids
        else _extract_meter_id(meter_name)
    )
    return WorkbookMetadata(
        meter_id=meter_id,
        meter_name=meter_name,
        object_name=object_name,
        device=device,
        period_start=period_start,
        period_end_exclusive=period_end_exclusive,
        summary_totals=summary_totals,
    )


def _find_hourly_header(frame: pd.DataFrame) -> tuple[int, int]:
    for row_index in range(len(frame)):
        values = [_clean_text(value).casefold() for value in frame.iloc[row_index]]
        for column_index, value in enumerate(values):
            if value == "дата" and {"с", "до"}.issubset(set(values)):
                return row_index, column_index
    raise AskueImportError("Hourly table header was not found")


def _meter_class(meter_name: str, role_by_name: dict[str, str] | None) -> str:
    role = (role_by_name or {}).get(meter_name, "outgoing_load")
    return {
        "master_input": "master_input",
        "reserve": "reserve",
        "outgoing_load": "university_load_candidate",
    }.get(role, "university_load_candidate")


def parse_workbook_frame(
    frame: pd.DataFrame,
    source_file: str,
    registry_ids: dict[str, str] | None = None,
    role_by_name: dict[str, str] | None = None,
    total_tolerance_kwh: float = 0.5,
) -> tuple[pd.DataFrame, WorkbookMetadata]:
    metadata = extract_metadata(frame, registry_ids=registry_ids)
    header_row, date_column = _find_hourly_header(frame)
    if date_column + 7 >= frame.shape[1]:
        raise AskueImportError("Hourly table does not contain all eight expected columns")

    hourly = frame.iloc[header_row + 1 :, date_column : date_column + 8].copy()
    hourly.columns = [
        "date",
        "start_time",
        "end_time",
        "active_import_kwh",
        "reactive_import_kvarh",
        "active_export_kwh",
        "reactive_export_kvarh",
        "reported_cos_phi",
    ]
    hourly = hourly[
        hourly["date"].notna()
        & hourly["start_time"].notna()
        & hourly["end_time"].notna()
    ].copy()
    if hourly.empty:
        raise AskueImportError("Hourly table contains no complete intervals")

    dates = hourly["date"].map(_parse_date_value)
    start_offsets: list[pd.Timedelta | pd.NaT] = []
    end_offsets: list[pd.Timedelta | pd.NaT] = []
    for start_value, end_value in zip(hourly["start_time"], hourly["end_time"]):
        try:
            start_offsets.append(_time_delta(start_value))
            end_offsets.append(_time_delta(end_value))
        except AskueImportError:
            start_offsets.append(pd.NaT)
            end_offsets.append(pd.NaT)
    hourly["timestamp_local"] = dates.dt.normalize() + pd.Series(
        start_offsets,
        index=hourly.index,
    )
    hourly["interval_end_offset"] = pd.Series(end_offsets, index=hourly.index)
    if hourly["timestamp_local"].isna().any() or hourly["interval_end_offset"].isna().any():
        bad_rows = (hourly.index[hourly["timestamp_local"].isna()] + 1).tolist()[:5]
        raise AskueImportError(f"Invalid timestamps in workbook rows {bad_rows}")

    energy_columns = [
        "active_import_kwh",
        "reactive_import_kvarh",
        "active_export_kwh",
        "reactive_export_kvarh",
    ]
    for column in energy_columns:
        converted = pd.to_numeric(hourly[column], errors="coerce")
        if converted.isna().any():
            bad_rows = (hourly.index[converted.isna()] + 1).tolist()[:5]
            raise AskueImportError(f"Invalid values in {column}, workbook rows {bad_rows}")
        hourly[column] = converted.astype(float)

    start_offset_series = pd.Series(start_offsets, index=hourly.index)
    duration_hours = (
        (hourly["interval_end_offset"] - start_offset_series)
        .dt.total_seconds()
        .div(3600)
    )
    duration_hours = duration_hours.where(duration_hours >= 0, duration_hours + 24)
    integral_duration = duration_hours.round().eq(duration_hours)
    if (~integral_duration | duration_hours.gt(24)).any():
        bad_rows = (hourly.index[~integral_duration | duration_hours.gt(24)] + 1).tolist()[:5]
        raise AskueImportError(f"Unsupported interval duration in workbook rows {bad_rows}")
    zero_duration_rows = int(duration_hours.eq(0).sum())
    split_multi_hour_rows = int(duration_hours.gt(1).sum())
    raw_negative = hourly[energy_columns].lt(0).any(axis=1)
    invalid_negative = raw_negative & duration_hours.ne(0)
    if invalid_negative.any():
        bad_rows = (hourly.index[invalid_negative] + 1).tolist()[:5]
        raise AskueImportError(
            f"Negative energy outside zero-duration adjustments in workbook rows {bad_rows}"
        )

    expanded_rows: list[dict[str, object]] = []
    for (_, row), raw_duration in zip(hourly.iterrows(), duration_hours):
        hours = int(raw_duration)
        expansion_count = max(hours, 1)
        flag = (
            "zero_duration_adjustment"
            if hours == 0
            else "split_multi_hour_interval"
            if hours > 1
            else ""
        )
        for hour_offset in range(expansion_count):
            expanded_rows.append({
                "timestamp_local": row["timestamp_local"] + pd.Timedelta(hours=hour_offset),
                **{
                    column: float(row[column]) / expansion_count
                    for column in energy_columns
                },
                "source_rows": 1,
                "quality_flags": flag,
            })
    hourly = pd.DataFrame(expanded_rows)
    hourly = (
        hourly.groupby("timestamp_local", as_index=False)
        .agg(
            active_import_kwh=("active_import_kwh", "sum"),
            reactive_import_kvarh=("reactive_import_kvarh", "sum"),
            active_export_kwh=("active_export_kwh", "sum"),
            reactive_export_kvarh=("reactive_export_kvarh", "sum"),
            source_rows=("source_rows", "sum"),
            quality_flags=(
                "quality_flags",
                lambda values: "|".join(sorted({value for value in values if value})),
            ),
        )
        .sort_values("timestamp_local")
    )
    aggregated_duplicate_rows = int((hourly["source_rows"] - 1).sum())
    if hourly[energy_columns].lt(0).any(axis=None):
        raise AskueImportError("Hourly totals remain negative after applying adjustments")
    expected_rows = int(
        (metadata.period_end_exclusive - metadata.period_start) / pd.Timedelta(hours=1)
    )
    if len(hourly) != expected_rows:
        raise AskueImportError(
            f"Expected {expected_rows} hourly rows for the report period, found {len(hourly)}"
        )
    expected_index = pd.date_range(
        metadata.period_start,
        metadata.period_end_exclusive,
        freq="h",
        inclusive="left",
    )
    if not hourly["timestamp_local"].reset_index(drop=True).equals(pd.Series(expected_index)):
        missing = expected_index.difference(hourly["timestamp_local"])
        raise AskueImportError(
            f"Hourly sequence is incomplete; first missing timestamp: {missing[0] if len(missing) else 'unknown'}"
        )

    for column, expected_total in zip(energy_columns, metadata.summary_totals):
        actual_total = float(hourly[column].sum())
        if not math.isclose(actual_total, expected_total, abs_tol=total_tolerance_kwh):
            raise AskueImportError(
                f"Summary mismatch for {column}: table={actual_total:.3f}, "
                f"report={expected_total:.3f}"
            )

    apparent = np.sqrt(
        hourly["active_import_kwh"] ** 2
        + hourly["reactive_import_kvarh"] ** 2
    )
    calculated_pf = np.where(
        apparent > 0,
        hourly["active_import_kwh"] / apparent,
        np.nan,
    )
    normalized = pd.DataFrame({
        "timestamp_local": hourly["timestamp_local"],
        "meter_id": metadata.meter_id,
        "meter_name": metadata.meter_name,
        "object": metadata.object_name,
        "device": metadata.device,
        "meter_class": _meter_class(metadata.meter_name, role_by_name),
        "active_import_kwh": hourly["active_import_kwh"],
        "reactive_import_kvarh": hourly["reactive_import_kvarh"],
        "active_export_kwh": hourly["active_export_kwh"],
        "reactive_export_kvarh": hourly["reactive_export_kvarh"],
        "cos_phi_calculated": calculated_pf,
        "source_rows": hourly["source_rows"],
        "quality_flags": hourly.apply(
            lambda row: "|".join(
                flag
                for flag in [
                    row["quality_flags"],
                    "multiple_source_rows" if row["source_rows"] > 1 else "",
                ]
                if flag
            ),
            axis=1,
        ),
        "source_file": source_file,
    })
    normalized = normalized[OUTPUT_COLUMNS]
    normalized.attrs["zero_duration_rows"] = zero_duration_rows
    normalized.attrs["split_multi_hour_rows"] = split_multi_hour_rows
    normalized.attrs["aggregated_duplicate_rows"] = aggregated_duplicate_rows
    return normalized, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(registry_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    registry = pd.read_csv(registry_path, dtype={"meter_id": str})
    required = {"meter_id", "meter_name", "role"}
    missing = required - set(registry.columns)
    if missing:
        raise AskueImportError(f"Meter registry is missing columns: {sorted(missing)}")
    if registry["meter_name"].duplicated().any():
        raise AskueImportError("Meter registry contains duplicate meter names")
    ids = dict(zip(registry["meter_name"], registry["meter_id"]))
    roles = dict(zip(registry["meter_name"], registry["role"]))
    return ids, roles


def import_askue_folder(
    input_dir: Path,
    output_path: Path,
    registry_path: Path,
    report_path: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise AskueImportError(f"ASKUE folder does not exist: {input_dir}")
    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls"}
    )
    if not files:
        raise AskueImportError(f"No Excel workbooks found in {input_dir}")

    registry_ids, roles = load_registry(registry_path)
    frames: list[pd.DataFrame] = []
    metadata_rows: list[WorkbookMetadata] = []
    file_rows: list[dict[str, Any]] = []
    for path in files:
        try:
            workbook_frame = pd.read_excel(path, sheet_name="часовой", header=None)
            normalized, metadata = parse_workbook_frame(
                workbook_frame,
                source_file=path.name,
                registry_ids=registry_ids,
                role_by_name=roles,
            )
        except Exception as exc:
            if isinstance(exc, AskueImportError):
                raise AskueImportError(f"{path.name}: {exc}") from exc
            raise AskueImportError(f"{path.name}: {type(exc).__name__}: {exc}") from exc
        frames.append(normalized)
        metadata_rows.append(metadata)
        file_rows.append({
            "file": path.name,
            "sha256": _sha256(path),
            "meter_id": metadata.meter_id,
            "meter_name": metadata.meter_name,
            "rows": len(normalized),
            "zero_duration_rows": int(normalized.attrs.get("zero_duration_rows", 0)),
            "split_multi_hour_rows": int(
                normalized.attrs.get("split_multi_hour_rows", 0)
            ),
            "aggregated_duplicate_rows": int(
                normalized.attrs.get("aggregated_duplicate_rows", 0)
            ),
            "active_import_kwh": float(normalized["active_import_kwh"].sum()),
        })

    meter_names = [metadata.meter_name for metadata in metadata_rows]
    duplicates = sorted({name for name in meter_names if meter_names.count(name) > 1})
    if duplicates:
        raise AskueImportError(f"Duplicate meter workbooks: {duplicates}")
    unknown = sorted(set(meter_names) - set(registry_ids))
    if unknown:
        raise AskueImportError(f"Meters absent from registry: {unknown}")
    missing = sorted(set(registry_ids) - set(meter_names))
    if missing and not allow_partial:
        raise AskueImportError(
            f"ASKUE folder is incomplete; missing {len(missing)} registry meters: {missing}"
        )
    periods = {
        (metadata.period_start, metadata.period_end_exclusive)
        for metadata in metadata_rows
    }
    if len(periods) != 1:
        raise AskueImportError("Workbooks do not cover the same reporting period")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["meter_id", "timestamp_local"]).reset_index(drop=True)
    if combined.duplicated(["meter_name", "timestamp_local"]).any():
        raise AskueImportError("Combined dataset contains duplicate meter timestamps")
    combined["timestamp_local"] = combined["timestamp_local"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".csv",
        prefix="hourly_energy_",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temporary_output = Path(handle.name)
        combined.to_csv(handle, index=False)
    try:
        os.replace(temporary_output, output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    period_start, period_end_exclusive = next(iter(periods))
    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "success",
        "input_directory": str(input_dir),
        "output_file": str(output_path.resolve()),
        "source_file_count": len(files),
        "meter_count": len(meter_names),
        "row_count": len(combined),
        "period_start": period_start.isoformat(),
        "period_end_exclusive": period_end_exclusive.isoformat(),
        "active_import_kwh": float(combined["active_import_kwh"].sum()),
        "missing_registry_meters": missing,
        "files": file_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
