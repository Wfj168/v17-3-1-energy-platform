from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_profiles_v17_3_1_tongjiang import generate_tongjiang_characteristic_profiles_v17_3_1
from model_core_v17_3_1_county import solve_integrated_energy_system_v17_3_1
from paper_run_v17_3_1_county import (
    N_STEPS_PER_HOUR,
    RESULT_ROOT,
    capacities_to_fixed_dict,
    get_table_value,
    save_result,
)
from scenarios_v17_3_1_county import get_reliability_stress_scenarios_v17_3_1


def _single_day(profiles: pd.DataFrame, day_type: str) -> pd.DataFrame:
    out = profiles[profiles["day_type"] == day_type].copy().reset_index(drop=True)
    out["day_id"] = 0
    out["day_weight"] = 1.0
    out["t_global"] = out.index * float(out["dt"].iloc[0])
    return out


def _scale_load(
    out: pd.DataFrame,
    electric: float = 1.0,
    heat: float = 1.0,
    cooling: float = 1.0,
) -> None:
    out["electric_load"] *= electric
    out["heat_load"] *= heat
    out["cooling_load"] *= cooling
    out["critical_electric_load"] *= electric
    out["critical_heat_load"] *= heat
    out["critical_cooling_load"] *= cooling


def build_reliability_cases(profiles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build one-day Tongjiang-characteristic reliability cases.

    Q0: normal spring irrigation day.
    Q1: moderate disturbance with higher irrigation demand and grid derating.
    Q2: moderate winter disturbance with low wind/PV and higher heating demand.
    Q3: severe compound disturbance with low renewables, high load and deep grid derating.
    """

    q0 = _single_day(profiles, "spring_irrigation")

    q1 = _single_day(profiles, "spring_irrigation")
    mask = (q1["hour"] >= 7.0) & (q1["hour"] < 19.0)
    q1.loc[mask, "grid_import_availability"] = 0.35
    _scale_load(q1, electric=1.08)

    q2 = _single_day(profiles, "winter_heating")
    mask = (q2["hour"] >= 16.0) & (q2["hour"] < 23.0)
    q2.loc[mask, "grid_import_availability"] = 0.45
    q2["pv_cf"] *= 0.50
    q2["wt_cf"] *= 0.60
    _scale_load(q2, electric=1.08, heat=1.10)

    q3 = _single_day(profiles, "winter_heating")
    mask = (q3["hour"] >= 12.0) & (q3["hour"] < 24.0)
    q3.loc[mask, "grid_import_availability"] = 0.25
    q3["pv_cf"] *= 0.40
    q3["wt_cf"] *= 0.40
    _scale_load(q3, electric=1.15, heat=1.12)

    return {"Q0": q0, "Q1": q1, "Q2": q2, "Q3": q3}


def _metric(result: dict, name: str) -> float:
    return get_table_value(result["metrics"], "Metric", name)


def _cost_item(result: dict, name: str) -> float:
    return get_table_value(result["cost"], "Cost item", name)


def _critical_load_service_rate(result: dict) -> float:
    """Return critical-load service rate using dispatch-level verification.

    The optimization only allows unmet demand within the non-critical share.
    This post-check quantifies whether any unmet demand exceeded that allowance.
    """

    d = result["dispatch"]
    dt = float(d["dt"].iloc[0])

    critical_total = 0.0
    critical_unmet = 0.0
    for load_col, critical_col, unmet_col in [
        ("electric_load", "critical_electric_load", "P_unmet_e"),
        ("heat_load", "critical_heat_load", "H_unmet"),
        ("cooling_load", "critical_cooling_load", "C_unmet"),
    ]:
        if critical_col not in d.columns:
            continue
        load = pd.to_numeric(d[load_col], errors="coerce").fillna(0.0).to_numpy(float)
        critical = pd.to_numeric(d[critical_col], errors="coerce").fillna(0.0).to_numpy(float)
        unmet = pd.to_numeric(d[unmet_col], errors="coerce").fillna(0.0).to_numpy(float)
        noncritical = np.maximum(0.0, load - critical)
        critical_total += float(np.sum(critical) * dt)
        critical_unmet += float(np.sum(np.maximum(0.0, unmet - noncritical)) * dt)

    if critical_total <= 1e-12:
        return 100.0
    return 100.0 * max(0.0, 1.0 - critical_unmet / critical_total)


def _reliability_cost_breakdown(result: dict) -> dict[str, float]:
    """Separate event-day operating cost from annual capacity-related charges.

    For one-day stress cases, most variable-cost rows are event-day values because
    day_weight=1. Grid demand charge and export-capacity charge remain annual
    capacity charges and must not be labelled EUR/day.
    """

    annual_demand_charge = _cost_item(result, "Grid demand charge")
    annual_export_capacity_charge = _cost_item(result, "Grid export capacity cost")
    generic_variable_cost = _cost_item(result, "Variable operating annual cost")
    carbon_external_cost = _cost_item(result, "Carbon external cost")

    event_day_private_cost = (
        generic_variable_cost
        - annual_demand_charge
        - annual_export_capacity_charge
    )
    event_day_social_cost = event_day_private_cost + carbon_external_cost

    return {
        "Event-day private operating cost [EUR/day]": event_day_private_cost,
        "Event-day carbon external cost [EUR/day]": carbon_external_cost,
        "Event-day social operating cost [EUR/day]": event_day_social_cost,
        "Annual grid demand charge [EUR/year]": annual_demand_charge,
        "Annual export capacity charge [EUR/year]": annual_export_capacity_charge,
    }


def run_reliability_v17_3_1() -> None:
    capacity_path = RESULT_ROOT / "planning" / "S4" / "data" / "capacity_results.csv"
    if not capacity_path.exists():
        raise FileNotFoundError("请先运行 paper_run_v17_3_1_county.py。")

    fixed_capacities = capacities_to_fixed_dict(pd.read_csv(capacity_path))
    full_profiles = generate_tongjiang_characteristic_profiles_v17_3_1(
        n_steps_per_hour=N_STEPS_PER_HOUR,
        seed=42,
    )
    cases = build_reliability_cases(full_profiles)
    scenarios = get_reliability_stress_scenarios_v17_3_1(fixed_capacities)

    rows: list[dict] = []
    for key in ["Q0", "Q1", "Q2", "Q3"]:
        print(f"Solving reliability case {key}: {scenarios[key]['name']}")
        result = solve_integrated_energy_system_v17_3_1(cases[key], scenarios[key])
        if not result["success"]:
            raise RuntimeError(f"{key} failed: {result['message']}")
        save_result(result, RESULT_ROOT / "reliability" / key / "data")

        row = {
            "Scenario": key,
            "Name": scenarios[key]["name"],
            "Day name": str(cases[key]["day_name"].iloc[0]),
            "Electric service rate [%]": _metric(result, "Electric service rate"),
            "Heat service rate [%]": _metric(result, "Heat service rate"),
            "Cooling service rate [%]": _metric(result, "Cooling service rate"),
            "Critical-load service rate [%]": _critical_load_service_rate(result),
            "Total multi-energy service rate [%]": _metric(result, "Total multi-energy service rate"),
            "Electric unmet [MWh/day]": _metric(result, "Annual electric unmet load"),
            "Heat unmet [MWh/day]": _metric(result, "Annual heat unmet load"),
            "Cooling unmet [MWh/day]": _metric(result, "Annual cooling unmet load"),
            "Total unmet [MWh/day]": _metric(result, "Annual unmet load"),
            "Shortage duration [h/day]": _metric(result, "Expected shortage duration"),
            "Peak grid import [MW]": _metric(result, "Peak grid import"),
        }
        row.update(_reliability_cost_breakdown(result))
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        RESULT_ROOT / "reliability_stress_summary_v17_3_1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # VOLL sensitivity on the severe disturbance. If shortage is physically
    # unavoidable, changing VOLL should mainly change economic loss rather than
    # eliminating the physical shortage.
    base = scenarios["Q3"]
    sensitivity_rows: list[dict] = []
    for multiplier in [0.5, 1.0, 2.0, 5.0]:
        scenario = dict(base)
        scenario["electric_unserved_energy_penalty"] *= multiplier
        scenario["heat_unserved_energy_penalty"] *= multiplier
        scenario["cooling_unserved_energy_penalty"] *= multiplier
        result = solve_integrated_energy_system_v17_3_1(cases["Q3"], scenario)
        if not result["success"]:
            sensitivity_rows.append({"VOLL multiplier": multiplier, "Success": False})
            continue

        row = {
            "VOLL multiplier": multiplier,
            "Success": True,
            "Total unmet [MWh/day]": _metric(result, "Annual unmet load"),
            "Total service rate [%]": _metric(result, "Total multi-energy service rate"),
            "Critical-load service rate [%]": _critical_load_service_rate(result),
        }
        row.update(_reliability_cost_breakdown(result))
        sensitivity_rows.append(row)

    pd.DataFrame(sensitivity_rows).to_csv(
        RESULT_ROOT / "reliability_voll_sensitivity_v17_3_1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("V17.3.1 reliability screening completed.")
    print(RESULT_ROOT.resolve())


if __name__ == "__main__":
    run_reliability_v17_3_1()
