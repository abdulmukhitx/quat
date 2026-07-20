from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .tariffs import TariffSchedule, apply_tariffs


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _finite_or_none(value: object) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    prepared = frame.copy()
    prepared = prepared.replace({np.nan: None})
    return prepared.to_dict(orient="records")


def build_portfolio_analysis(
    data: pd.DataFrame,
    registry: pd.DataFrame,
    schedule: TariffSchedule,
    model_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = data.copy()
    frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"])
    frame["active_import_kwh"] = pd.to_numeric(
        frame["active_import_kwh"], errors="coerce"
    ).fillna(0.0)
    frame["reactive_import_kvarh"] = pd.to_numeric(
        frame.get("reactive_import_kvarh"), errors="coerce"
    )
    frame["cos_phi_calculated"] = pd.to_numeric(
        frame.get("cos_phi_calculated"), errors="coerce"
    )

    registry_columns = [
        "meter_id",
        "meter_name",
        "role",
        "include_in_load_model",
        "university_owned",
        "university_pays",
        "university_controls",
    ]
    registry_view = registry[registry_columns].copy().rename(
        columns={"meter_id": "registry_meter_id"}
    )
    registry_view["include_in_load_model"] = truthy(
        registry_view["include_in_load_model"]
    )
    merged = frame.merge(
        registry_view,
        on="meter_name",
        how="left",
        validate="many_to_one",
        suffixes=("", "_registry"),
    )
    missing_registry = sorted(
        merged.loc[merged["role"].isna(), "meter_name"].dropna().unique()
    )
    merged["month"] = merged["timestamp_local"].dt.to_period("M").astype(str)

    master = merged[merged["role"] == "master_input"].copy()
    if master.empty:
        raise ValueError("No master input meters are defined in the registry")
    master = apply_tariffs(master, schedule)

    master_monthly_long = (
        master.groupby(["month", "registry_meter_id"], as_index=False)
        .agg(
            energy_kwh=("active_import_kwh", "sum"),
            cost_kzt=("energy_cost_kzt", "sum"),
        )
    )
    energy_pivot = master_monthly_long.pivot(
        index="month", columns="registry_meter_id", values="energy_kwh"
    ).fillna(0.0)
    cost_monthly = master.groupby("month")["energy_cost_kzt"].sum()
    rate_monthly = (
        master.groupby("month")["energy_cost_kzt"].sum()
        / master.groupby("month")["active_import_kwh"].sum().replace(0, np.nan)
    )
    months = sorted(master["month"].unique())
    master_monthly_rows: list[dict[str, Any]] = []
    for month in months:
        t1 = float(energy_pivot.loc[month].get("4", 0.0))
        t2 = float(energy_pivot.loc[month].get("15", 0.0))
        total = t1 + t2
        master_monthly_rows.append({
            "month": month,
            "input_t1_kwh": t1,
            "input_t2_kwh": t2,
            "total_input_kwh": total,
            "gross_cost_kzt": float(cost_monthly.get(month, 0.0)),
            "weighted_tariff_kzt_per_kwh": _finite_or_none(
                rate_monthly.get(month, np.nan)
            ),
        })

    master_by_input = (
        master.groupby(["registry_meter_id", "meter_name"], as_index=False)
        .agg(
            annual_energy_kwh=("active_import_kwh", "sum"),
            annual_cost_kzt=("energy_cost_kzt", "sum"),
            zero_hour_share=("active_import_kwh", lambda values: float((values == 0).mean())),
            peak_hourly_kwh=("active_import_kwh", "max"),
        )
        .sort_values("annual_energy_kwh", ascending=False)
        .rename(columns={"registry_meter_id": "meter_id"})
    )

    modeled = merged[merged["include_in_load_model"].fillna(False)].copy()
    modeled = apply_tariffs(modeled, schedule)
    modeled["is_night"] = (
        (modeled["timestamp_local"].dt.hour < 6)
        | (modeled["timestamp_local"].dt.hour >= 22)
    )
    modeled["is_weekend"] = modeled["timestamp_local"].dt.dayofweek >= 5
    modeled["is_off_hours"] = (
        modeled["is_weekend"]
        | (modeled["timestamp_local"].dt.hour < 7)
        | (modeled["timestamp_local"].dt.hour >= 20)
    )

    best_models: dict[str, dict[str, Any]] = {}
    event_by_meter: dict[str, dict[str, float | int]] = {}
    if model_analysis:
        for row in model_analysis.get("summary", {}).get("best_models", []):
            best_models[str(row["meter_name"])] = row
        events = pd.DataFrame(model_analysis.get("events", []))
        if not events.empty:
            event_summary = (
                events.groupby("meter_name", as_index=False)
                .agg(
                    candidate_event_count=("meter_name", "size"),
                    candidate_excess_kwh=("excess_kwh", "sum"),
                    estimated_candidate_excess_cost_kzt=(
                        "estimated_excess_cost_kzt", "sum"
                    ),
                )
            )
            event_by_meter = {
                str(row["meter_name"]): row
                for row in event_summary.to_dict(orient="records")
            }

    priced_all = apply_tariffs(merged, schedule)
    priced_all["is_off_hours"] = (
        (priced_all["timestamp_local"].dt.dayofweek >= 5)
        | (priced_all["timestamp_local"].dt.hour < 7)
        | (priced_all["timestamp_local"].dt.hour >= 20)
    )
    all_meter_rows: list[dict[str, Any]] = []
    for meter_name, group in priced_all[priced_all["role"].notna()].groupby(
        "meter_name", sort=True
    ):
        active = group["active_import_kwh"]
        annual = float(active.sum())
        valid_pf = active.gt(0) & group["cos_phi_calculated"].notna()
        weighted_pf = (
            float(
                np.average(
                    group.loc[valid_pf, "cos_phi_calculated"],
                    weights=active.loc[valid_pf],
                )
            )
            if valid_pf.any() and active.loc[valid_pf].sum() > 0
            else None
        )
        best = best_models.get(str(meter_name), {})
        event = event_by_meter.get(str(meter_name), {})
        all_meter_rows.append({
            "meter_id": str(group["registry_meter_id"].iloc[0]),
            "meter_name": str(meter_name),
            "role": str(group["role"].iloc[0]),
            "include_in_load_model": bool(group["include_in_load_model"].iloc[0]),
            "annual_energy_kwh": annual,
            "annual_reactive_import_kvarh": float(
                group["reactive_import_kvarh"].fillna(0).sum()
            ),
            "annual_active_export_kwh": float(
                pd.to_numeric(group.get("active_export_kwh"), errors="coerce")
                .fillna(0)
                .sum()
            ),
            "equivalent_gross_cost_kzt": float(group["energy_cost_kzt"].sum()),
            "average_hourly_kwh": float(active.mean()),
            "peak_hourly_kwh": float(active.max()),
            "zero_hour_share": float(active.eq(0).mean()),
            "off_hours_energy_share": (
                float(group.loc[group["is_off_hours"], "active_import_kwh"].sum() / annual)
                if annual else None
            ),
            "weighted_power_factor": weighted_pf,
            "best_baseline_model": best.get("model"),
            "validation_wmape": best.get("wmape"),
            "candidate_event_count": int(event.get("candidate_event_count", 0)),
            "candidate_excess_kwh": float(event.get("candidate_excess_kwh", 0.0)),
            "estimated_candidate_excess_cost_kzt": float(
                event.get("estimated_candidate_excess_cost_kzt", 0.0)
            ),
        })
    all_meter_rows.sort(key=lambda row: row["annual_energy_kwh"], reverse=True)
    monthly_all_meters = (
        priced_all[priced_all["role"].notna()]
        .groupby(["month", "registry_meter_id", "meter_name"], as_index=False)
        .agg(
            energy_kwh=("active_import_kwh", "sum"),
            equivalent_gross_cost_kzt=("energy_cost_kzt", "sum"),
        )
        .sort_values(["month", "energy_kwh"], ascending=[True, False])
        .rename(columns={"registry_meter_id": "meter_id"})
    )

    meter_rows: list[dict[str, Any]] = []
    for meter_name, group in modeled.groupby("meter_name", sort=True):
        active = group["active_import_kwh"]
        annual = float(active.sum())
        valid_pf = (
            active.gt(0)
            & group["cos_phi_calculated"].notna()
        )
        weighted_pf = (
            float(
                np.average(
                    group.loc[valid_pf, "cos_phi_calculated"],
                    weights=active.loc[valid_pf],
                )
            )
            if valid_pf.any() and active.loc[valid_pf].sum() > 0
            else None
        )
        best = best_models.get(str(meter_name), {})
        event = event_by_meter.get(str(meter_name), {})
        meter_rows.append({
            "meter_id": str(group["registry_meter_id"].iloc[0]),
            "meter_name": str(meter_name),
            "annual_energy_kwh": annual,
            "equivalent_gross_cost_kzt": float(group["energy_cost_kzt"].sum()),
            "share_of_modeled_load": 0.0,
            "average_hourly_kwh": float(active.mean()),
            "p10_hourly_kwh": float(active.quantile(0.10)),
            "p95_hourly_kwh": float(active.quantile(0.95)),
            "peak_hourly_kwh": float(active.max()),
            "load_factor": float(active.mean() / active.max()) if active.max() else None,
            "zero_hour_share": float(active.eq(0).mean()),
            "night_energy_share": float(
                group.loc[group["is_night"], "active_import_kwh"].sum() / annual
            ) if annual else None,
            "weekend_energy_share": float(
                group.loc[group["is_weekend"], "active_import_kwh"].sum() / annual
            ) if annual else None,
            "off_hours_energy_share": float(
                group.loc[group["is_off_hours"], "active_import_kwh"].sum() / annual
            ) if annual else None,
            "weighted_power_factor": weighted_pf,
            "low_pf_active_hour_share": float(
                (group.loc[valid_pf, "cos_phi_calculated"] < 0.9).mean()
            ) if valid_pf.any() else None,
            "best_baseline_model": best.get("model"),
            "validation_wmape": best.get("wmape"),
            "candidate_event_count": int(event.get("candidate_event_count", 0)),
            "candidate_excess_kwh": float(event.get("candidate_excess_kwh", 0.0)),
            "estimated_candidate_excess_cost_kzt": float(
                event.get("estimated_candidate_excess_cost_kzt", 0.0)
            ),
        })
    modeled_total = sum(row["annual_energy_kwh"] for row in meter_rows)
    for row in meter_rows:
        row["share_of_modeled_load"] = (
            row["annual_energy_kwh"] / modeled_total if modeled_total else 0.0
        )
    meter_rows.sort(key=lambda row: row["annual_energy_kwh"], reverse=True)

    modeled_monthly = (
        modeled.groupby(["month", "registry_meter_id", "meter_name"], as_index=False)
        .agg(
            energy_kwh=("active_import_kwh", "sum"),
            equivalent_gross_cost_kzt=("energy_cost_kzt", "sum"),
        )
        .sort_values(["month", "energy_kwh"], ascending=[True, False])
        .rename(columns={"registry_meter_id": "meter_id"})
    )

    zero_master_months = [
        {
            "month": row["month"],
            "meter_id": str(row["registry_meter_id"]),
            "meter_name": row["meter_name"],
        }
        for row in (
            master.assign(
                month=master["timestamp_local"].dt.to_period("M").astype(str)
            )
            .groupby(["month", "registry_meter_id", "meter_name"], as_index=False)[
                "active_import_kwh"
            ]
            .sum()
            .query("active_import_kwh == 0")
            .to_dict(orient="records")
        )
    ]

    total_input = float(master["active_import_kwh"].sum())
    outgoing_difference = modeled_total - total_input
    return {
        "summary": {
            "period_start": master["timestamp_local"].min().isoformat(),
            "period_end_exclusive": (
                master["timestamp_local"].max() + pd.Timedelta(hours=1)
            ).isoformat(),
            "consumer_group": schedule.consumer_group,
            "master_input_energy_kwh": total_input,
            "estimated_gross_cost_kzt": float(master["energy_cost_kzt"].sum()),
            "weighted_tariff_kzt_per_kwh": float(
                master["energy_cost_kzt"].sum() / total_input
            ),
            "master_input_meter_count": int(master["meter_name"].nunique()),
            "modeled_load_meter_count": int(modeled["meter_name"].nunique()),
            "modeled_load_energy_kwh": modeled_total,
            "unvalidated_outgoing_minus_input_kwh": outgoing_difference,
            "unvalidated_outgoing_minus_input_pct": (
                outgoing_difference / total_input if total_input else None
            ),
            "candidate_event_count": int(
                model_analysis.get("summary", {}).get("candidate_event_count", 0)
                if model_analysis else 0
            ),
            "candidate_excess_kwh": float(
                model_analysis.get("summary", {}).get("candidate_excess_kwh", 0.0)
                if model_analysis else 0.0
            ),
            "estimated_candidate_excess_cost_kzt": float(
                model_analysis.get("summary", {}).get(
                    "estimated_candidate_excess_cost_kzt", 0.0
                ) if model_analysis else 0.0
            ),
        },
        "monthly_master_inputs": master_monthly_rows,
        "master_inputs": _records(master_by_input),
        "all_meters": all_meter_rows,
        "monthly_all_meters": _records(monthly_all_meters),
        "modeled_loads": meter_rows,
        "monthly_modeled_loads": _records(modeled_monthly),
        "quality": {
            "missing_registry_meters": missing_registry,
            "zero_master_input_months": zero_master_months,
            "topology_balance_status": "unvalidated_pending_single_line_diagram",
            "topology_note": (
                "Outgoing cells are not treated as an accounting balance until "
                "their association with input bus sections is confirmed."
            ),
        },
    }
