from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class TariffPeriod:
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    rate_with_vat_kzt_per_kwh: float
    status: str


@dataclass(frozen=True)
class TariffSchedule:
    consumer_group: int
    periods: tuple[TariffPeriod, ...]
    currency: str = "KZT"

    def rate_at(self, timestamp: object) -> TariffPeriod:
        moment = pd.Timestamp(timestamp)
        for period in self.periods:
            if period.start <= moment < period.end_exclusive:
                return period
        raise ValueError(f"No tariff covers timestamp {moment.isoformat()}")


def schedule_from_payload(
    payload: dict[str, Any],
    consumer_group: int | None = None,
) -> TariffSchedule:
    group = int(
        payload.get("selected_consumer_group", 2)
        if consumer_group is None
        else consumer_group
    )
    group_key = f"group_{group}"
    periods: list[TariffPeriod] = []
    for item in payload.get("periods", []):
        if group_key not in item:
            raise ValueError(f"Consumer group {group} is absent from a tariff period")
        group_values = item[group_key]
        rate = group_values.get("rate_with_vat_kzt_per_kwh")
        if rate is None:
            raise ValueError(
                f"Gross tariff is missing for group {group} at {item.get('start')}"
            )
        periods.append(
            TariffPeriod(
                start=pd.Timestamp(item["start"]),
                end_exclusive=pd.Timestamp(item["end_exclusive"]),
                rate_with_vat_kzt_per_kwh=float(rate),
                status=str(group_values.get("status", "unspecified")),
            )
        )
    if not periods:
        raise ValueError("Tariff schedule is empty")
    periods.sort(key=lambda period: period.start)
    for previous, current in zip(periods, periods[1:]):
        if previous.end_exclusive > current.start:
            raise ValueError("Tariff periods overlap")
    return TariffSchedule(
        consumer_group=group,
        periods=tuple(periods),
        currency=str(payload.get("currency", "KZT")),
    )


def load_tariff_schedule(
    path: str | Path,
    consumer_group: int | None = None,
) -> TariffSchedule:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return schedule_from_payload(payload, consumer_group=consumer_group)


def apply_tariffs(
    frame: pd.DataFrame,
    schedule: TariffSchedule,
    *,
    timestamp_column: str = "timestamp_local",
    energy_column: str = "active_import_kwh",
    strict: bool = True,
) -> pd.DataFrame:
    """Attach gross tariff and energy cost without changing source rows."""
    result = frame.copy()
    timestamps = pd.to_datetime(result[timestamp_column])
    result["tariff_rate_kzt_per_kwh"] = pd.NA
    result["tariff_status"] = pd.NA
    for period in schedule.periods:
        mask = (
            (timestamps >= period.start)
            & (timestamps < period.end_exclusive)
        )
        result.loc[mask, "tariff_rate_kzt_per_kwh"] = (
            period.rate_with_vat_kzt_per_kwh
        )
        result.loc[mask, "tariff_status"] = period.status
    result["tariff_rate_kzt_per_kwh"] = pd.to_numeric(
        result["tariff_rate_kzt_per_kwh"], errors="coerce"
    )
    if strict and result["tariff_rate_kzt_per_kwh"].isna().any():
        uncovered = timestamps[result["tariff_rate_kzt_per_kwh"].isna()]
        raise ValueError(
            "Tariff schedule does not cover all rows: "
            f"{uncovered.min().isoformat()} to {uncovered.max().isoformat()}"
        )
    result["energy_cost_kzt"] = (
        pd.to_numeric(result[energy_column], errors="coerce")
        * result["tariff_rate_kzt_per_kwh"]
    )
    return result


def price_candidate_events(
    events: pd.DataFrame,
    schedule: TariffSchedule,
) -> pd.DataFrame:
    """Estimate the gross value of candidate excess energy at each event start."""
    if events.empty:
        return events.copy()
    result = events.copy()
    priced = [schedule.rate_at(timestamp) for timestamp in result["start"]]
    result["tariff_rate_kzt_per_kwh"] = [
        period.rate_with_vat_kzt_per_kwh for period in priced
    ]
    result["tariff_status"] = [period.status for period in priced]
    result["estimated_excess_cost_kzt"] = (
        pd.to_numeric(result["excess_kwh"], errors="coerce")
        * result["tariff_rate_kzt_per_kwh"]
    )
    return result


def validate_contiguous_coverage(
    periods: Iterable[TariffPeriod],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return gaps between adjacent periods for quality-control reporting."""
    ordered = sorted(periods, key=lambda period: period.start)
    return [
        (previous.end_exclusive, current.start)
        for previous, current in zip(ordered, ordered[1:])
        if previous.end_exclusive < current.start
    ]
