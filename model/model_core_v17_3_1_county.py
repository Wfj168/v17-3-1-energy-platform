from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack


def calculate_crf(discount_rate: float, life_years: int) -> float:
    if life_years <= 0:
        raise ValueError("life_years must be positive")
    if abs(discount_rate) < 1e-12:
        return 1.0 / life_years
    r = discount_rate
    n = life_years
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def build_sparse_matrix(rows_data, n_rows: int, n_cols: int):
    if n_rows == 0:
        return None
    rows, cols, data = [], [], []
    for r, entries in enumerate(rows_data):
        for c, value in entries:
            if abs(value) > 1e-12:
                rows.append(r)
                cols.append(c)
                data.append(value)
    return coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()


def infer_dt(profiles: pd.DataFrame) -> float:
    if "dt" in profiles.columns:
        return float(pd.to_numeric(profiles["dt"], errors="coerce").dropna().median())
    if "hour" in profiles.columns and len(profiles) >= 2:
        values = pd.to_numeric(profiles["hour"], errors="coerce").dropna().values
        if len(values) >= 2:
            return float(np.median(np.diff(values)))
    return 1.0


def get_day_groups(profiles: pd.DataFrame) -> dict[Any, list[int]]:
    if "day_id" not in profiles.columns:
        return {0: list(range(len(profiles)))}
    return {
        day_id: list(sub.index)
        for day_id, sub in profiles.groupby("day_id", sort=True)
    }


def get_annual_days(profiles: pd.DataFrame) -> float:
    if "day_id" in profiles.columns and "day_weight" in profiles.columns:
        return float(profiles.groupby("day_id")["day_weight"].first().sum())
    return 365.0


def get_weight_array(profiles: pd.DataFrame) -> np.ndarray:
    if "day_weight" in profiles.columns:
        return pd.to_numeric(profiles["day_weight"], errors="coerce").fillna(365.0).values
    return np.full(len(profiles), 365.0)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0 or np.sum(weights) <= 0:
        return 0.0
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def solve_integrated_energy_system_v17_3_1(
    profiles: pd.DataFrame,
    scenario: dict,
) -> dict:
    """County-scale low-carbon planning-operation co-optimization model.

    V17.3.1 improvements over V17.1:
    1. Retains county-scale planning-operation co-optimization and Tongjiang-characteristic representative days.
    2. Adds grid import/export exclusivity and prohibits fossil-dominated export accounting.
    3. Adds export wheeling and export-capacity charges to avoid an export-led plan.
    4. Adds annual storage-cycle limits and explicit flexible-load in/out overlap limits.
    5. Supports deterministic rule-dispatch profiles for a fair R0 baseline.
    6. Reports renewable local absorption/export/curtailment with consistent denominators.
    7. Adds grid-exchange, demand-response and storage-operation diagnostics.
    8. Keeps carbon cap, carbon external cost, private cost and social cost separated.
    9. Replaces hard-zero unmet demand with critical-load protection, carrier-specific VOLL and an annual EENS limit.
    10. Supports time-varying grid availability and Tongjiang-characteristic reliability stress tests.
    11. Separates annualized capital cost, fixed O&M, variable operating cost, private total cost and social total cost.
    """

    required_columns = {
        "electric_load",
        "heat_load",
        "cooling_load",
        "pv_cf",
        "wt_cf",
        "gas_price",
        "carbon_price",
        "hour",
    }
    missing = sorted(required_columns - set(profiles.columns))
    if missing:
        raise ValueError(f"profiles missing required columns: {missing}")

    T = len(profiles)
    dt = infer_dt(profiles)
    weights = get_weight_array(profiles)
    annual_days = get_annual_days(profiles)
    day_groups = get_day_groups(profiles)

    buy_price = (
        profiles["electricity_buy_price"].values
        if "electricity_buy_price" in profiles.columns
        else profiles["electricity_price"].values
    )
    sell_price = (
        profiles["electricity_sell_price"].values
        if "electricity_sell_price" in profiles.columns
        else np.zeros(T)
    )

    grid_import_availability = (
        pd.to_numeric(profiles["grid_import_availability"], errors="coerce")
        .fillna(1.0).clip(0.0, 1.0).values
        if "grid_import_availability" in profiles.columns
        else np.ones(T)
    )
    critical_electric_load = (
        profiles["critical_electric_load"].values
        if "critical_electric_load" in profiles.columns
        else 0.70 * profiles["electric_load"].values
    )
    critical_heat_load = (
        profiles["critical_heat_load"].values
        if "critical_heat_load" in profiles.columns
        else 0.85 * profiles["heat_load"].values
    )
    critical_cooling_load = (
        profiles["critical_cooling_load"].values
        if "critical_cooling_load" in profiles.columns
        else 0.60 * profiles["cooling_load"].values
    )
    noncritical_electric_load = np.maximum(
        0.0, profiles["electric_load"].values - critical_electric_load
    )
    noncritical_heat_load = np.maximum(
        0.0, profiles["heat_load"].values - critical_heat_load
    )
    noncritical_cooling_load = np.maximum(
        0.0, profiles["cooling_load"].values - critical_cooling_load
    )

    # -----------------------------
    # Technical and economic inputs
    # -----------------------------
    discount_rate = float(scenario.get("discount_rate", 0.05))
    life_years = int(scenario.get("life_years", 20))
    fixed_om_rate = float(scenario.get("fixed_om_rate", 0.02))
    crf = calculate_crf(discount_rate, life_years)

    eta_gb = float(scenario.get("eta_gb", 0.90))
    cop_hp = float(scenario.get("cop_hp", 3.2))
    cop_ec = float(scenario.get("cop_ec", 3.0))
    eta_chp_e = float(scenario.get("eta_chp_e", 0.35))
    eta_chp_h = float(scenario.get("eta_chp_h", 0.45))

    eta_bat_ch = float(scenario.get("eta_bat_ch", 0.95))
    eta_bat_dis = float(scenario.get("eta_bat_dis", 0.95))
    soc_bat_min = float(scenario.get("soc_bat_min", 0.10))
    soc_bat_max = float(scenario.get("soc_bat_max", 0.90))
    bat_initial_soc = float(scenario.get("bat_initial_soc", 0.50))
    bat_self_discharge_per_hour = float(scenario.get("bat_self_discharge_per_hour", 0.0002))
    bat_power_energy_ratio = float(scenario.get("bat_power_energy_ratio", 0.50))

    eta_ts_ch = float(scenario.get("eta_ts_ch", 0.95))
    eta_ts_dis = float(scenario.get("eta_ts_dis", 0.95))
    soc_ts_min = float(scenario.get("soc_ts_min", 0.10))
    soc_ts_max = float(scenario.get("soc_ts_max", 0.90))
    ts_initial_soc = float(scenario.get("ts_initial_soc", 0.50))
    ts_self_discharge_per_hour = float(scenario.get("ts_self_discharge_per_hour", 0.0020))
    ts_power_energy_ratio = float(scenario.get("ts_power_energy_ratio", 0.50))

    enable_carbon_constraint = bool(scenario.get("enable_carbon_constraint", False))
    internalize_carbon_price = bool(scenario.get("internalize_carbon_price", False))
    enable_battery = bool(scenario.get("enable_battery", False))
    enable_thermal_storage = bool(scenario.get("enable_thermal_storage", False))
    battery_dispatch_enabled = bool(scenario.get("battery_dispatch_enabled", enable_battery))
    thermal_storage_dispatch_enabled = bool(
        scenario.get("thermal_storage_dispatch_enabled", enable_thermal_storage)
    )
    enable_flexible_load = bool(scenario.get("enable_flexible_load", False))
    enable_interruptible_load = bool(scenario.get("enable_interruptible_load", False))
    enable_chp = bool(scenario.get("enable_chp", True))
    enable_grid_export = bool(scenario.get("enable_grid_export", True))
    force_zero_unmet = bool(scenario.get("force_zero_unmet", True))
    strict_storage_exclusivity = bool(scenario.get("strict_storage_exclusivity", False))
    strict_grid_exchange_exclusivity = bool(
        scenario.get("strict_grid_exchange_exclusivity", False)
    )
    storage_dispatch_mode = str(scenario.get("storage_dispatch_mode", "optimized")).strip().lower()
    if storage_dispatch_mode not in {"optimized", "rule", "disabled"}:
        raise ValueError("storage_dispatch_mode must be optimized, rule or disabled")
    if storage_dispatch_mode == "disabled":
        battery_dispatch_enabled = False
        thermal_storage_dispatch_enabled = False
    max_annual_export_share = scenario.get("max_annual_export_share_of_renewables", 0.15)
    if max_annual_export_share is not None:
        max_annual_export_share = float(max_annual_export_share)
        if not 0.0 <= max_annual_export_share <= 1.0:
            raise ValueError("max_annual_export_share_of_renewables must be between 0 and 1")
    max_annual_export_mwh = scenario.get("max_annual_export_mwh", None)


    grid_emission_factor = float(scenario.get("grid_emission_factor", 0.55))
    gas_emission_factor = float(scenario.get("gas_emission_factor", 0.20))

    renewable_curtailment_penalty = float(scenario.get("renewable_curtailment_penalty", 20.0))
    load_shifting_cost = float(scenario.get("load_shifting_cost", 30.0))
    interruptible_load_cost = float(scenario.get("interruptible_load_cost", 1200.0))
    fallback_voll = float(scenario.get("unmet_load_penalty", 15000.0))
    electric_unserved_energy_penalty = float(
        scenario.get("electric_unserved_energy_penalty", fallback_voll)
    )
    heat_unserved_energy_penalty = float(
        scenario.get("heat_unserved_energy_penalty", 0.67 * fallback_voll)
    )
    cooling_unserved_energy_penalty = float(
        scenario.get("cooling_unserved_energy_penalty", 0.56 * fallback_voll)
    )
    max_annual_unserved_energy_ratio = scenario.get(
        "max_annual_unserved_energy_ratio", 0.0001
    )
    if max_annual_unserved_energy_ratio is not None:
        max_annual_unserved_energy_ratio = float(max_annual_unserved_energy_ratio)
        if not 0.0 <= max_annual_unserved_energy_ratio <= 1.0:
            raise ValueError("max_annual_unserved_energy_ratio must be between 0 and 1")
    battery_degradation_cost = float(scenario.get("battery_degradation_cost", 14.0))
    thermal_storage_degradation_cost = float(scenario.get("thermal_storage_degradation_cost", 2.0))
    grid_demand_charge = float(scenario.get("grid_demand_charge", 18000.0))
    grid_export_capacity_charge = float(
        scenario.get("grid_export_capacity_charge", 6000.0)
    )
    grid_export_wheeling_charge = float(
        scenario.get("grid_export_wheeling_charge", 8.0)
    )
    grid_smoothing_penalty = float(scenario.get("grid_smoothing_penalty", 8.0))
    grid_ramp_limit_per_hour = float(scenario.get("grid_ramp_limit_per_hour", 30.0))
    device_smoothing_penalty = float(scenario.get("device_smoothing_penalty", 5.0))

    grid_import_capacity = float(scenario.get("grid_import_capacity", 120.0))
    grid_export_capacity = float(scenario.get("grid_export_capacity", 5.0)) if enable_grid_export else 0.0
    reserve_margin = float(scenario.get("reserve_margin", 0.10))
    enforce_planning_adequacy = bool(scenario.get("enforce_planning_adequacy", True))
    pv_capacity_credit = float(scenario.get("pv_capacity_credit", 0.08))
    wt_capacity_credit = float(scenario.get("wt_capacity_credit", 0.15))
    max_daily_shift_energy_ratio = float(scenario.get("max_daily_shift_energy_ratio", 0.06))
    max_renewable_energy_to_service_demand_ratio = scenario.get(
        "max_renewable_energy_to_service_demand_ratio", 1.20
    )
    if max_renewable_energy_to_service_demand_ratio is not None:
        max_renewable_energy_to_service_demand_ratio = float(
            max_renewable_energy_to_service_demand_ratio
        )
    max_battery_equivalent_cycles = scenario.get(
        "max_battery_equivalent_cycles", 350.0
    )
    max_thermal_storage_equivalent_cycles = scenario.get(
        "max_thermal_storage_equivalent_cycles", 365.0
    )
    if max_battery_equivalent_cycles is not None:
        max_battery_equivalent_cycles = float(max_battery_equivalent_cycles)
    if max_thermal_storage_equivalent_cycles is not None:
        max_thermal_storage_equivalent_cycles = float(
            max_thermal_storage_equivalent_cycles
        )

    chp_ramp_rate_per_hour = float(scenario.get("chp_ramp_rate_per_hour", 0.45))
    hp_ramp_rate_per_hour = float(scenario.get("hp_ramp_rate_per_hour", 0.80))
    ec_ramp_rate_per_hour = float(scenario.get("ec_ramp_rate_per_hour", 0.80))
    gb_ramp_rate_per_hour = float(scenario.get("gb_ramp_rate_per_hour", 0.70))

    max_capacity = {
        "C_PV": float(scenario.get("max_pv_capacity", 240.0)),
        "C_WT": float(scenario.get("max_wt_capacity", 240.0)),
        "C_CHP": float(scenario.get("max_chp_capacity", 100.0)) if enable_chp else 0.0,
        "C_HP": float(scenario.get("max_hp_capacity", 150.0)),
        "C_EC": float(scenario.get("max_ec_capacity", 120.0)),
        "C_GB": float(scenario.get("max_gb_capacity", 150.0)),
        "C_BAT_E": float(scenario.get("max_battery_energy_capacity", 300.0)) if enable_battery else 0.0,
        "C_BAT_P": float(scenario.get("max_battery_power_capacity", 120.0)) if enable_battery else 0.0,
        "C_TS_E": float(scenario.get("max_thermal_storage_energy_capacity", 300.0)) if enable_thermal_storage else 0.0,
        "C_TS_P": float(scenario.get("max_thermal_storage_power_capacity", 120.0)) if enable_thermal_storage else 0.0,
    }

    capex = {
        "C_PV": float(scenario.get("capex_pv", 600000.0)),
        "C_WT": float(scenario.get("capex_wt", 1100000.0)),
        "C_CHP": float(scenario.get("capex_chp", 576923.0)),
        "C_HP": float(scenario.get("capex_hp", 250000.0)),
        "C_EC": float(scenario.get("capex_ec", 180000.0)),
        "C_GB": float(scenario.get("capex_gb", 120000.0)),
        "C_BAT_E": float(scenario.get("capex_bat_e", 250000.0)),
        "C_BAT_P": float(scenario.get("capex_bat_p", 100000.0)),
        "C_TS_E": float(scenario.get("capex_ts_e", 50000.0)),
        "C_TS_P": float(scenario.get("capex_ts_p", 30000.0)),
    }
    annualized_capital_cost_per_unit = {
        key: value * crf for key, value in capex.items()
    }
    annual_fixed_om_cost_per_unit = {
        key: value * fixed_om_rate for key, value in capex.items()
    }
    annual_capacity_cost = {
        key: annualized_capital_cost_per_unit[key] + annual_fixed_om_cost_per_unit[key]
        for key in capex
    }

    fixed_capacities = scenario.get("fixed_capacities", {}) or {}

    # -----------------------------
    # Variable registry
    # -----------------------------
    idx: dict[str, Any] = {}
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    objective: list[float] = []
    integrality: list[int] = []

    def add_scalar(
        name: str,
        lower: float = 0.0,
        upper: float | None = None,
        obj: float = 0.0,
        integer: bool = False,
    ) -> None:
        idx[name] = len(lower_bounds)
        lower_bounds.append(float(lower))
        upper_bounds.append(np.inf if upper is None else float(upper))
        objective.append(float(obj))
        integrality.append(1 if integer else 0)

    def add_vector(
        name: str,
        size: int,
        lower: float = 0.0,
        upper: float | None = None,
        obj=None,
        integer: bool = False,
    ) -> None:
        start = len(lower_bounds)
        idx[name] = slice(start, start + size)
        if obj is None:
            obj_values = np.zeros(size)
        elif np.isscalar(obj):
            obj_values = np.full(size, float(obj))
        else:
            obj_values = np.asarray(obj, dtype=float)
            if len(obj_values) != size:
                raise ValueError(f"objective length mismatch for {name}")
        ub_value = np.inf if upper is None else float(upper)
        for i in range(size):
            lower_bounds.append(float(lower))
            upper_bounds.append(ub_value)
            objective.append(float(obj_values[i]))
            integrality.append(1 if integer else 0)

    def v(name: str, t: int | None = None) -> int:
        item = idx[name]
        if isinstance(item, slice):
            if t is None:
                raise ValueError(f"time index required for vector {name}")
            return item.start + t
        return int(item)

    # Planning capacities. Fixed-capacity mode is implemented with equal bounds.
    for cap_name in [
        "C_PV",
        "C_WT",
        "C_CHP",
        "C_HP",
        "C_EC",
        "C_GB",
        "C_BAT_E",
        "C_BAT_P",
        "C_TS_E",
        "C_TS_P",
    ]:
        if cap_name in fixed_capacities:
            fixed_value = float(fixed_capacities[cap_name])
            add_scalar(cap_name, fixed_value, fixed_value, annual_capacity_cost[cap_name])
        else:
            add_scalar(cap_name, 0.0, max_capacity[cap_name], annual_capacity_cost[cap_name])

    add_scalar("P_grid_peak", 0.0, grid_import_capacity, grid_demand_charge)
    add_scalar(
        "P_grid_export_peak",
        0.0,
        grid_export_capacity,
        grid_export_capacity_charge,
    )

    # Operating variables
    add_vector("P_grid_buy", T, upper=grid_import_capacity, obj=buy_price * dt * weights)
    add_vector(
        "P_grid_sell",
        T,
        upper=grid_export_capacity,
        obj=(-sell_price + grid_export_wheeling_charge) * dt * weights,
    )
    add_vector("P_PV", T)
    add_vector("P_WT", T)
    add_vector("P_PV_curt", T, obj=renewable_curtailment_penalty * dt * weights)
    add_vector("P_WT_curt", T, obj=renewable_curtailment_penalty * dt * weights)

    add_vector("G_CHP", T)
    add_vector("P_CHP", T)
    add_vector("H_CHP", T)
    add_vector("P_HP", T)
    add_vector("H_HP", T)
    add_vector("P_EC", T)
    add_vector("C_EC_out", T)
    add_vector("G_GB", T)
    add_vector("H_GB", T)

    battery_dispatch_upper = None if battery_dispatch_enabled else 0.0
    thermal_dispatch_upper = None if thermal_storage_dispatch_enabled else 0.0
    add_vector(
        "P_BAT_ch",
        T,
        upper=battery_dispatch_upper,
        obj=0.5 * battery_degradation_cost * dt * weights,
    )
    add_vector(
        "P_BAT_dis",
        T,
        upper=battery_dispatch_upper,
        obj=0.5 * battery_degradation_cost * dt * weights,
    )
    add_vector("SOC_BAT", T)

    add_vector(
        "H_TS_ch",
        T,
        upper=thermal_dispatch_upper,
        obj=0.5 * thermal_storage_degradation_cost * dt * weights,
    )
    add_vector(
        "H_TS_dis",
        T,
        upper=thermal_dispatch_upper,
        obj=0.5 * thermal_storage_degradation_cost * dt * weights,
    )
    add_vector("SOC_TS", T)

    if strict_storage_exclusivity and battery_dispatch_enabled:
        add_vector("U_BAT_CH", T, lower=0.0, upper=1.0, integer=True)
    if strict_storage_exclusivity and thermal_storage_dispatch_enabled:
        add_vector("U_TS_CH", T, lower=0.0, upper=1.0, integer=True)
    if strict_grid_exchange_exclusivity and enable_grid_export:
        add_vector("U_GRID_IMPORT", T, lower=0.0, upper=1.0, integer=True)

    flex_upper = None if enable_flexible_load else 0.0
    interrupt_upper = None if enable_interruptible_load else 0.0
    add_vector(
        "P_shift_in",
        T,
        upper=flex_upper,
        obj=0.5 * load_shifting_cost * dt * weights,
    )
    add_vector(
        "P_shift_out",
        T,
        upper=flex_upper,
        obj=0.5 * load_shifting_cost * dt * weights,
    )
    add_vector(
        "P_interrupt",
        T,
        upper=interrupt_upper,
        obj=interruptible_load_cost * dt * weights,
    )

    unmet_upper = 0.0 if force_zero_unmet else None
    add_vector(
        "P_unmet_e", T, upper=unmet_upper,
        obj=electric_unserved_energy_penalty * dt * weights,
    )
    add_vector(
        "H_unmet", T, upper=unmet_upper,
        obj=heat_unserved_energy_penalty * dt * weights,
    )
    add_vector(
        "C_unmet", T, upper=unmet_upper,
        obj=cooling_unserved_energy_penalty * dt * weights,
    )

    # Optional deterministic rule-dispatch profiles.  These are used by R0/R2
    # to create a true non-optimized baseline rather than an optimization that
    # only restricts charge/discharge time windows.
    fixed_profiles = {
        "P_BAT_ch": scenario.get("fixed_battery_charge_profile"),
        "P_BAT_dis": scenario.get("fixed_battery_discharge_profile"),
        "H_TS_ch": scenario.get("fixed_thermal_charge_profile"),
        "H_TS_dis": scenario.get("fixed_thermal_discharge_profile"),
    }
    for variable_name, profile_values in fixed_profiles.items():
        if profile_values is None:
            continue
        values = np.asarray(profile_values, dtype=float)
        if len(values) != T:
            raise ValueError(
                f"{variable_name} fixed profile length {len(values)} != {T}"
            )
        values = np.maximum(values, 0.0)
        sl = idx[variable_name]
        for t, value in enumerate(values):
            lower_bounds[sl.start + t] = float(value)
            upper_bounds[sl.start + t] = float(value)

    # Absolute-change variables for device and grid smoothing.
    for prefix in ["CHP", "HP", "EC", "GB"]:
        add_vector(f"D_{prefix}_up", T, obj=device_smoothing_penalty * weights)
        add_vector(f"D_{prefix}_down", T, obj=device_smoothing_penalty * weights)
    add_vector("D_GRID_up", T, obj=grid_smoothing_penalty * weights)
    add_vector("D_GRID_down", T, obj=grid_smoothing_penalty * weights)

    # Fuel and carbon objective terms.
    for t in range(T):
        gas_price_t = float(profiles["gas_price"].iloc[t])
        carbon_price_t = float(profiles["carbon_price"].iloc[t])
        objective[v("G_GB", t)] += gas_price_t * dt * weights[t]
        objective[v("G_CHP", t)] += gas_price_t * dt * weights[t]
        if internalize_carbon_price:
            objective[v("P_grid_buy", t)] += (
                grid_emission_factor * carbon_price_t * dt * weights[t]
            )
            objective[v("G_GB", t)] += (
                gas_emission_factor * carbon_price_t * dt * weights[t]
            )
            objective[v("G_CHP", t)] += (
                gas_emission_factor * carbon_price_t * dt * weights[t]
            )

    n_var = len(lower_bounds)
    eq_rows, eq_b = [], []
    ub_rows, ub_b = [], []

    # Renewable availability
    for t in range(T):
        eq_rows.append(
            [
                (v("P_PV", t), 1.0),
                (v("P_PV_curt", t), 1.0),
                (v("C_PV"), -float(profiles["pv_cf"].iloc[t])),
            ]
        )
        eq_b.append(0.0)
        eq_rows.append(
            [
                (v("P_WT", t), 1.0),
                (v("P_WT_curt", t), 1.0),
                (v("C_WT"), -float(profiles["wt_cf"].iloc[t])),
            ]
        )
        eq_b.append(0.0)

    # Conversion relations
    for t in range(T):
        eq_rows.append([(v("P_CHP", t), 1.0), (v("G_CHP", t), -eta_chp_e)])
        eq_b.append(0.0)
        eq_rows.append([(v("H_CHP", t), 1.0), (v("G_CHP", t), -eta_chp_h)])
        eq_b.append(0.0)
        eq_rows.append([(v("H_HP", t), 1.0), (v("P_HP", t), -cop_hp)])
        eq_b.append(0.0)
        eq_rows.append([(v("C_EC_out", t), 1.0), (v("P_EC", t), -cop_ec)])
        eq_b.append(0.0)
        eq_rows.append([(v("H_GB", t), 1.0), (v("G_GB", t), -eta_gb)])
        eq_b.append(0.0)

    # Electric, heat and cooling balances
    for t in range(T):
        eq_rows.append(
            [
                (v("P_grid_buy", t), 1.0),
                (v("P_grid_sell", t), -1.0),
                (v("P_PV", t), 1.0),
                (v("P_WT", t), 1.0),
                (v("P_CHP", t), 1.0),
                (v("P_BAT_dis", t), 1.0),
                (v("P_unmet_e", t), 1.0),
                (v("P_BAT_ch", t), -1.0),
                (v("P_HP", t), -1.0),
                (v("P_EC", t), -1.0),
                (v("P_shift_in", t), -1.0),
                (v("P_shift_out", t), 1.0),
                (v("P_interrupt", t), 1.0),
            ]
        )
        eq_b.append(float(profiles["electric_load"].iloc[t]))

        eq_rows.append(
            [
                (v("H_CHP", t), 1.0),
                (v("H_GB", t), 1.0),
                (v("H_HP", t), 1.0),
                (v("H_TS_dis", t), 1.0),
                (v("H_unmet", t), 1.0),
                (v("H_TS_ch", t), -1.0),
            ]
        )
        eq_b.append(float(profiles["heat_load"].iloc[t]))

        eq_rows.append([(v("C_EC_out", t), 1.0), (v("C_unmet", t), 1.0)])
        eq_b.append(float(profiles["cooling_load"].iloc[t]))

    # Equipment capacities, grid demand and storage power.
    for t in range(T):
        ub_rows.append([(v("P_CHP", t), 1.0), (v("C_CHP"), -1.0)])
        ub_b.append(0.0)
        ub_rows.append([(v("H_HP", t), 1.0), (v("C_HP"), -1.0)])
        ub_b.append(0.0)
        ub_rows.append([(v("C_EC_out", t), 1.0), (v("C_EC"), -1.0)])
        ub_b.append(0.0)
        ub_rows.append([(v("H_GB", t), 1.0), (v("C_GB"), -1.0)])
        ub_b.append(0.0)
        ub_rows.append([(v("P_BAT_ch", t), 1.0), (v("C_BAT_P"), -1.0)])
        ub_b.append(0.0)
        ub_rows.append([(v("P_BAT_dis", t), 1.0), (v("C_BAT_P"), -1.0)])
        ub_b.append(0.0)
        ub_rows.append([(v("H_TS_ch", t), 1.0), (v("C_TS_P"), -1.0)])
        ub_b.append(0.0)
        ub_rows.append([(v("H_TS_dis", t), 1.0), (v("C_TS_P"), -1.0)])
        ub_b.append(0.0)
        ub_rows.append([(v("P_grid_buy", t), 1.0), (v("P_grid_peak"), -1.0)])
        ub_b.append(0.0)
        ub_rows.append([(v("P_grid_sell", t), 1.0), (v("P_grid_export_peak"), -1.0)])
        ub_b.append(0.0)

        # Time-varying grid availability is used for normal operation and
        # Tongjiang-characteristic contingency screening with the same model.
        ub_rows.append([(v("P_grid_buy", t), 1.0)])
        ub_b.append(grid_import_capacity * float(grid_import_availability[t]))

        # Critical demand is protected.  When shortage variables are enabled,
        # only the non-critical portion may be unserved.
        ub_rows.append([(v("P_unmet_e", t), 1.0)])
        ub_b.append(float(noncritical_electric_load[t]))
        ub_rows.append([(v("H_unmet", t), 1.0)])
        ub_b.append(float(noncritical_heat_load[t]))
        ub_rows.append([(v("C_unmet", t), 1.0)])
        ub_b.append(float(noncritical_cooling_load[t]))

        # Export is limited by contemporaneous renewable generation.  Combined
        # with import/export exclusivity this prevents artificial grid arbitrage
        # and avoids attributing fossil generation to renewable export.
        ub_rows.append([
            (v("P_grid_sell", t), 1.0),
            (v("P_PV", t), -1.0),
            (v("P_WT", t), -1.0),
        ])
        ub_b.append(0.0)

        if strict_grid_exchange_exclusivity and enable_grid_export:
            ub_rows.append([
                (v("P_grid_buy", t), 1.0),
                (v("U_GRID_IMPORT", t), -grid_import_capacity),
            ])
            ub_b.append(0.0)
            ub_rows.append([
                (v("P_grid_sell", t), 1.0),
                (v("U_GRID_IMPORT", t), grid_export_capacity),
            ])
            ub_b.append(grid_export_capacity)

        if strict_storage_exclusivity and battery_dispatch_enabled:
            big_m = max_capacity["C_BAT_P"]
            ub_rows.append([(v("P_BAT_ch", t), 1.0), (v("U_BAT_CH", t), -big_m)])
            ub_b.append(0.0)
            ub_rows.append([(v("P_BAT_dis", t), 1.0), (v("U_BAT_CH", t), big_m)])
            ub_b.append(big_m)

        if strict_storage_exclusivity and thermal_storage_dispatch_enabled:
            big_m = max_capacity["C_TS_P"]
            ub_rows.append([(v("H_TS_ch", t), 1.0), (v("U_TS_CH", t), -big_m)])
            ub_b.append(0.0)
            ub_rows.append([(v("H_TS_dis", t), 1.0), (v("U_TS_CH", t), big_m)])
            ub_b.append(big_m)

    # Conventional rule-constrained storage operation for R0/R2.
    def hour_in_windows(hour_value: float, windows) -> bool:
        return any(float(start) <= hour_value < float(end) for start, end in windows)

    if storage_dispatch_mode == "rule":
        bat_charge_windows = scenario.get("battery_rule_charge_windows", [(0.0, 6.5), (10.0, 15.5)])
        bat_discharge_windows = scenario.get("battery_rule_discharge_windows", [(7.0, 10.0), (17.0, 22.0)])
        ts_charge_windows = scenario.get("thermal_rule_charge_windows", [(0.0, 6.5), (10.0, 16.0)])
        ts_discharge_windows = scenario.get("thermal_rule_discharge_windows", [(6.5, 10.0), (17.0, 23.0)])
        for t in range(T):
            hour_t = float(profiles["hour"].iloc[t])
            if battery_dispatch_enabled and not hour_in_windows(hour_t, bat_charge_windows):
                ub_rows.append([(v("P_BAT_ch", t), 1.0)])
                ub_b.append(0.0)
            if battery_dispatch_enabled and not hour_in_windows(hour_t, bat_discharge_windows):
                ub_rows.append([(v("P_BAT_dis", t), 1.0)])
                ub_b.append(0.0)
            if thermal_storage_dispatch_enabled and not hour_in_windows(hour_t, ts_charge_windows):
                ub_rows.append([(v("H_TS_ch", t), 1.0)])
                ub_b.append(0.0)
            if thermal_storage_dispatch_enabled and not hour_in_windows(hour_t, ts_discharge_windows):
                ub_rows.append([(v("H_TS_dis", t), 1.0)])
                ub_b.append(0.0)

    ub_rows.append([(v("C_BAT_P"), 1.0), (v("C_BAT_E"), -bat_power_energy_ratio)])
    ub_b.append(0.0)
    ub_rows.append([(v("C_TS_P"), 1.0), (v("C_TS_E"), -ts_power_energy_ratio)])
    ub_b.append(0.0)

    def add_ramp_and_smooth(
        y_name: str,
        cap_name: str,
        prefix: str,
        ramp_rate_per_hour: float,
    ) -> None:
        for times in day_groups.values():
            for k, t in enumerate(times):
                prev = times[k - 1]
                ub_rows.append(
                    [
                        (v(y_name, t), 1.0),
                        (v(y_name, prev), -1.0),
                        (v(cap_name), -ramp_rate_per_hour * dt),
                    ]
                )
                ub_b.append(0.0)
                ub_rows.append(
                    [
                        (v(y_name, prev), 1.0),
                        (v(y_name, t), -1.0),
                        (v(cap_name), -ramp_rate_per_hour * dt),
                    ]
                )
                ub_b.append(0.0)
                eq_rows.append(
                    [
                        (v(y_name, t), 1.0),
                        (v(y_name, prev), -1.0),
                        (v(f"D_{prefix}_up", t), -1.0),
                        (v(f"D_{prefix}_down", t), 1.0),
                    ]
                )
                eq_b.append(0.0)

    add_ramp_and_smooth("P_CHP", "C_CHP", "CHP", chp_ramp_rate_per_hour)
    add_ramp_and_smooth("H_HP", "C_HP", "HP", hp_ramp_rate_per_hour)
    add_ramp_and_smooth("C_EC_out", "C_EC", "EC", ec_ramp_rate_per_hour)
    add_ramp_and_smooth("H_GB", "C_GB", "GB", gb_ramp_rate_per_hour)

    # Grid net-power smoothing: net grid = import - export.
    for times in day_groups.values():
        for k, t in enumerate(times):
            prev = times[k - 1]
            eq_rows.append(
                [
                    (v("P_grid_buy", t), 1.0),
                    (v("P_grid_sell", t), -1.0),
                    (v("P_grid_buy", prev), -1.0),
                    (v("P_grid_sell", prev), 1.0),
                    (v("D_GRID_up", t), -1.0),
                    (v("D_GRID_down", t), 1.0),
                ]
            )
            eq_b.append(0.0)
            if grid_ramp_limit_per_hour > 0:
                ramp_limit = grid_ramp_limit_per_hour * dt
                ub_rows.append([
                    (v("P_grid_buy", t), 1.0),
                    (v("P_grid_sell", t), -1.0),
                    (v("P_grid_buy", prev), -1.0),
                    (v("P_grid_sell", prev), 1.0),
                ])
                ub_b.append(ramp_limit)
                ub_rows.append([
                    (v("P_grid_buy", prev), 1.0),
                    (v("P_grid_sell", prev), -1.0),
                    (v("P_grid_buy", t), -1.0),
                    (v("P_grid_sell", t), 1.0),
                ])
                ub_b.append(ramp_limit)

    # Storage SOC with self-discharge and daily cyclic closure.
    for times in day_groups.values():
        for k, t in enumerate(times):
            retention = 1.0 if not battery_dispatch_enabled else max(0.0, 1.0 - bat_self_discharge_per_hour * dt)
            if k == 0:
                eq_rows.append(
                    [
                        (v("SOC_BAT", t), 1.0),
                        (v("P_BAT_ch", t), -eta_bat_ch * dt),
                        (v("P_BAT_dis", t), dt / eta_bat_dis),
                        (v("C_BAT_E"), -retention * bat_initial_soc),
                    ]
                )
            else:
                prev = times[k - 1]
                eq_rows.append(
                    [
                        (v("SOC_BAT", t), 1.0),
                        (v("SOC_BAT", prev), -retention),
                        (v("P_BAT_ch", t), -eta_bat_ch * dt),
                        (v("P_BAT_dis", t), dt / eta_bat_dis),
                    ]
                )
            eq_b.append(0.0)
            ub_rows.append([(v("SOC_BAT", t), 1.0), (v("C_BAT_E"), -soc_bat_max)])
            ub_b.append(0.0)
            ub_rows.append([(v("SOC_BAT", t), -1.0), (v("C_BAT_E"), soc_bat_min)])
            ub_b.append(0.0)
        eq_rows.append([(v("SOC_BAT", times[-1]), 1.0), (v("C_BAT_E"), -bat_initial_soc)])
        eq_b.append(0.0)

    for times in day_groups.values():
        for k, t in enumerate(times):
            retention = 1.0 if not thermal_storage_dispatch_enabled else max(0.0, 1.0 - ts_self_discharge_per_hour * dt)
            if k == 0:
                eq_rows.append(
                    [
                        (v("SOC_TS", t), 1.0),
                        (v("H_TS_ch", t), -eta_ts_ch * dt),
                        (v("H_TS_dis", t), dt / eta_ts_dis),
                        (v("C_TS_E"), -retention * ts_initial_soc),
                    ]
                )
            else:
                prev = times[k - 1]
                eq_rows.append(
                    [
                        (v("SOC_TS", t), 1.0),
                        (v("SOC_TS", prev), -retention),
                        (v("H_TS_ch", t), -eta_ts_ch * dt),
                        (v("H_TS_dis", t), dt / eta_ts_dis),
                    ]
                )
            eq_b.append(0.0)
            ub_rows.append([(v("SOC_TS", t), 1.0), (v("C_TS_E"), -soc_ts_max)])
            ub_b.append(0.0)
            ub_rows.append([(v("SOC_TS", t), -1.0), (v("C_TS_E"), soc_ts_min)])
            ub_b.append(0.0)
        eq_rows.append([(v("SOC_TS", times[-1]), 1.0), (v("C_TS_E"), -ts_initial_soc)])
        eq_b.append(0.0)

    # Flexible load: energy-neutral shifting within each representative day.
    if enable_flexible_load:
        shift_multiplier = float(scenario.get("shiftable_load_multiplier", 1.0))
        for times in day_groups.values():
            entries = []
            for t in times:
                entries.append((v("P_shift_in", t), dt))
                entries.append((v("P_shift_out", t), -dt))
            eq_rows.append(entries)
            eq_b.append(0.0)

            daily_energy = float((profiles.loc[times, "electric_load"] * dt).sum())
            ub_rows.append([(v("P_shift_out", t), dt) for t in times])
            ub_b.append(max_daily_shift_energy_ratio * daily_energy)

        shift_out_base = (
            profiles["shiftable_load_base"].values
            if "shiftable_load_base" in profiles.columns
            else 0.10 * profiles["electric_load"].values
        )
        shift_in_base = (
            profiles["shift_in_capacity"].values
            if "shift_in_capacity" in profiles.columns
            else 0.10 * profiles["electric_load"].values
        )
        for t in range(T):
            shift_out_limit = float(shift_multiplier * shift_out_base[t])
            shift_in_limit = float(shift_multiplier * shift_in_base[t])
            ub_rows.append([(v("P_shift_out", t), 1.0)])
            ub_b.append(shift_out_limit)
            ub_rows.append([(v("P_shift_in", t), 1.0)])
            ub_b.append(shift_in_limit)
            # Prevent simultaneous large upward and downward response.
            ub_rows.append([
                (v("P_shift_out", t), 1.0),
                (v("P_shift_in", t), 1.0),
            ])
            ub_b.append(max(shift_out_limit, shift_in_limit))

    if enable_interruptible_load:
        interrupt_base = (
            profiles["interruptible_load_base"].values
            if "interruptible_load_base" in profiles.columns
            else np.zeros(T)
        )
        interruptible_ratio = float(scenario.get("interruptible_load_ratio", 0.25))
        for t in range(T):
            ub_rows.append([(v("P_interrupt", t), 1.0)])
            ub_b.append(float(interruptible_ratio * interrupt_base[t]))

    # Annual expected energy-not-served (EENS) limit for normal planning and
    # operation. Stress tests can disable this limit to reveal physical shortages.
    if (not force_zero_unmet) and max_annual_unserved_energy_ratio is not None:
        total_annual_demand = float(
            (
                profiles["electric_load"].values
                + profiles["heat_load"].values
                + profiles["cooling_load"].values
            ).dot(dt * weights)
        )
        entries = []
        for t in range(T):
            coeff = dt * weights[t]
            entries.extend([
                (v("P_unmet_e", t), coeff),
                (v("H_unmet", t), coeff),
                (v("C_unmet", t), coeff),
            ])
        ub_rows.append(entries)
        ub_b.append(max_annual_unserved_energy_ratio * total_annual_demand)

    # Reserve-margin constraints excluding storage, so fixed-capacity no-storage
    # operation scenarios remain physically feasible.
    if enforce_planning_adequacy:
        max_cooling = float(profiles["cooling_load"].max())
        ub_rows.append([(v("C_EC"), -1.0)])
        ub_b.append(-(1.0 + reserve_margin) * max_cooling)

        max_heat = float(profiles["heat_load"].max())
        heat_ratio_chp = eta_chp_h / max(eta_chp_e, 1e-9)
        ub_rows.append(
            [
                (v("C_HP"), -1.0),
                (v("C_CHP"), -heat_ratio_chp),
                (v("C_GB"), -1.0),
            ]
        )
        ub_b.append(-(1.0 + reserve_margin) * max_heat)

        electric_service_proxy = (
            profiles["electric_load"]
            + profiles["heat_load"] / cop_hp
            + profiles["cooling_load"] / cop_ec
        )
        required_local_firm = (1.0 + reserve_margin) * float(electric_service_proxy.max()) - grid_import_capacity
        if required_local_firm > 0:
            ub_rows.append(
                [
                    (v("C_CHP"), -1.0),
                    (v("C_PV"), -pv_capacity_credit),
                    (v("C_WT"), -wt_capacity_credit),
                ]
            )
            ub_b.append(-required_local_firm)

    # Local-consumption-oriented renewable planning envelope.  The available
    # annual renewable energy is limited relative to the county's aggregated
    # electric-service demand (direct electricity plus heat/cooling electricity
    # equivalents).  This prevents an unrealistic export-led overbuild.
    if max_renewable_energy_to_service_demand_ratio is not None:
        pv_available_coeff = float((profiles["pv_cf"].values * dt * weights).sum())
        wt_available_coeff = float((profiles["wt_cf"].values * dt * weights).sum())
        annual_service_demand_proxy = float(
            (
                profiles["electric_load"].values
                + profiles["heat_load"].values / cop_hp
                + profiles["cooling_load"].values / cop_ec
            )
            .dot(dt * weights)
        )
        ub_rows.append([
            (v("C_PV"), pv_available_coeff),
            (v("C_WT"), wt_available_coeff),
        ])
        ub_b.append(
            max_renewable_energy_to_service_demand_ratio
            * annual_service_demand_proxy
        )

    # Annual storage-throughput constraints.  They limit excessive cycling and
    # make the linear degradation-cost representation more physically credible.
    if enable_battery and max_battery_equivalent_cycles is not None:
        entries = []
        for t in range(T):
            entries.append((v("P_BAT_ch", t), 0.5 * dt * weights[t]))
            entries.append((v("P_BAT_dis", t), 0.5 * dt * weights[t]))
        entries.append((v("C_BAT_E"), -max_battery_equivalent_cycles))
        ub_rows.append(entries)
        ub_b.append(0.0)
    if enable_thermal_storage and max_thermal_storage_equivalent_cycles is not None:
        entries = []
        for t in range(T):
            entries.append((v("H_TS_ch", t), 0.5 * dt * weights[t]))
            entries.append((v("H_TS_dis", t), 0.5 * dt * weights[t]))
        entries.append((v("C_TS_E"), -max_thermal_storage_equivalent_cycles))
        ub_rows.append(entries)
        ub_b.append(0.0)

    # Annual export restrictions. These prevent an unrealistically export-led plan.
    if enable_grid_export and max_annual_export_share is not None:
        pv_available_coeff = float((profiles["pv_cf"].values * dt * weights).sum())
        wt_available_coeff = float((profiles["wt_cf"].values * dt * weights).sum())
        entries = [(v("P_grid_sell", t), dt * weights[t]) for t in range(T)]
        entries.extend([
            (v("C_PV"), -max_annual_export_share * pv_available_coeff),
            (v("C_WT"), -max_annual_export_share * wt_available_coeff),
        ])
        ub_rows.append(entries)
        ub_b.append(0.0)
    if enable_grid_export and max_annual_export_mwh is not None:
        ub_rows.append([(v("P_grid_sell", t), dt * weights[t]) for t in range(T)])
        ub_b.append(float(max_annual_export_mwh))

    # Annual carbon cap.
    if enable_carbon_constraint:
        entries = []
        for t in range(T):
            entries.extend(
                [
                    (v("P_grid_buy", t), grid_emission_factor * dt * weights[t]),
                    (v("G_GB", t), gas_emission_factor * dt * weights[t]),
                    (v("G_CHP", t), gas_emission_factor * dt * weights[t]),
                ]
            )
        ub_rows.append(entries)
        ub_b.append(float(scenario.get("co2_cap", 1e12)) * annual_days)

    A_eq = build_sparse_matrix(eq_rows, len(eq_rows), n_var)
    A_ub = build_sparse_matrix(ub_rows, len(ub_rows), n_var)

    c_array = np.asarray(objective, dtype=float)
    lb_array = np.asarray(lower_bounds, dtype=float)
    ub_array = np.asarray(upper_bounds, dtype=float)
    integrality_array = np.asarray(integrality, dtype=int)

    if np.any(integrality_array != 0):
        constraints = []
        if A_eq is not None:
            b_eq_array = np.asarray(eq_b, dtype=float)
            constraints.append(LinearConstraint(A_eq, b_eq_array, b_eq_array))
        if A_ub is not None:
            constraints.append(
                LinearConstraint(
                    A_ub,
                    np.full(len(ub_b), -np.inf),
                    np.asarray(ub_b, dtype=float),
                )
            )
        result = milp(
            c=c_array,
            integrality=integrality_array,
            bounds=Bounds(lb_array, ub_array),
            constraints=constraints,
            options={
                "time_limit": float(scenario.get("milp_time_limit", 300.0)),
                "mip_rel_gap": float(scenario.get("mip_rel_gap", 0.002)),
                "disp": bool(scenario.get("solver_display", False)),
            },
        )
    else:
        result = linprog(
            c=c_array,
            A_ub=A_ub,
            b_ub=np.asarray(ub_b, dtype=float),
            A_eq=A_eq,
            b_eq=np.asarray(eq_b, dtype=float),
            bounds=list(zip(lb_array, ub_array)),
            method="highs",
        )

    if not result.success or result.x is None:
        return {
            "success": False,
            "message": str(result.message),
            "solver_status": int(result.status),
        }

    x = np.asarray(result.x, dtype=float)

    # -----------------------------
    # Results and diagnostics
    # -----------------------------
    technology_names = [
        "PV",
        "WT",
        "CHP",
        "Heat pump",
        "Electric chiller",
        "Gas boiler",
        "Battery energy",
        "Battery power",
        "Thermal storage energy",
        "Thermal storage power",
    ]
    capacity_keys = [
        "C_PV",
        "C_WT",
        "C_CHP",
        "C_HP",
        "C_EC",
        "C_GB",
        "C_BAT_E",
        "C_BAT_P",
        "C_TS_E",
        "C_TS_P",
    ]
    capacity_units = ["MW", "MW", "MW", "MW", "MW", "MW", "MWh", "MW", "MWh", "MW"]
    capacity_values = [x[v(key)] for key in capacity_keys]
    capacity_df = pd.DataFrame(
        {
            "Technology": technology_names,
            "Capacity": capacity_values,
            "Unit": capacity_units,
            "Model key": capacity_keys,
            "Upper bound": [max_capacity[key] for key in capacity_keys],
        }
    )
    capacity_df["Upper-bound utilization [%]"] = np.where(
        capacity_df["Upper bound"] > 1e-9,
        100.0 * capacity_df["Capacity"] / capacity_df["Upper bound"],
        0.0,
    )
    capacity_df["Upper bound binding"] = capacity_df["Upper-bound utilization [%]"] >= 95.0

    dispatch = pd.DataFrame(
        {
            "day_id": profiles.get("day_id", pd.Series(np.zeros(T, dtype=int))),
            "day_type": profiles.get("day_type", pd.Series(["single"] * T)),
            "day_name": profiles.get("day_name", profiles.get("day_type", pd.Series(["single"] * T))),
            "day_weight": weights,
            "hour": profiles["hour"].values,
            "dt": dt,
            "electric_load": profiles["electric_load"].values,
            "heat_load": profiles["heat_load"].values,
            "cooling_load": profiles["cooling_load"].values,
            "P_grid_buy": x[idx["P_grid_buy"]],
            "P_grid_sell": x[idx["P_grid_sell"]],
            "P_grid_net": x[idx["P_grid_buy"]] - x[idx["P_grid_sell"]],
            "P_PV": x[idx["P_PV"]],
            "P_WT": x[idx["P_WT"]],
            "P_PV_curt": x[idx["P_PV_curt"]],
            "P_WT_curt": x[idx["P_WT_curt"]],
            "G_CHP": x[idx["G_CHP"]],
            "P_CHP": x[idx["P_CHP"]],
            "H_CHP": x[idx["H_CHP"]],
            "P_HP": x[idx["P_HP"]],
            "H_HP": x[idx["H_HP"]],
            "P_EC": x[idx["P_EC"]],
            "C_EC_out": x[idx["C_EC_out"]],
            "G_GB": x[idx["G_GB"]],
            "H_GB": x[idx["H_GB"]],
            "P_BAT_ch": x[idx["P_BAT_ch"]],
            "P_BAT_dis": x[idx["P_BAT_dis"]],
            "SOC_BAT": x[idx["SOC_BAT"]],
            "H_TS_ch": x[idx["H_TS_ch"]],
            "H_TS_dis": x[idx["H_TS_dis"]],
            "SOC_TS": x[idx["SOC_TS"]],
            "P_shift_in": x[idx["P_shift_in"]],
            "P_shift_out": x[idx["P_shift_out"]],
            "P_interrupt": x[idx["P_interrupt"]],
            "P_unmet_e": x[idx["P_unmet_e"]],
            "H_unmet": x[idx["H_unmet"]],
            "C_unmet": x[idx["C_unmet"]],
        }
    )

    for optional_col in [
        "residential_load",
        "commercial_load",
        "industrial_load",
        "cold_storage_load",
        "agro_processing_load",
        "irrigation_load",
        "electricity_buy_price",
        "electricity_sell_price",
        "gas_price",
        "carbon_price",
        "pv_cf",
        "wt_cf",
        "grid_import_availability",
        "critical_electric_load",
        "critical_heat_load",
        "critical_cooling_load",
    ]:
        if optional_col in profiles.columns:
            dispatch[optional_col] = profiles[optional_col].values

    dispatch["electric_load_adjusted"] = (
        dispatch["electric_load"]
        + dispatch["P_shift_in"]
        - dispatch["P_shift_out"]
        - dispatch["P_interrupt"]
    )
    dispatch["renewable_available"] = (
        x[v("C_PV")] * profiles["pv_cf"].values
        + x[v("C_WT")] * profiles["wt_cf"].values
    )
    dispatch["renewable_used"] = dispatch["P_PV"] + dispatch["P_WT"]
    dispatch["renewable_curtailment"] = dispatch["P_PV_curt"] + dispatch["P_WT_curt"]
    dispatch["co2_emission_rate"] = (
        grid_emission_factor * dispatch["P_grid_buy"]
        + gas_emission_factor * (dispatch["G_CHP"] + dispatch["G_GB"])
    )
    dispatch["co2_emission"] = dispatch["co2_emission_rate"] * dt
    dispatch["co2_emission_annual"] = dispatch["co2_emission"] * dispatch["day_weight"]
    dispatch["battery_simultaneous_overlap"] = np.minimum(
        dispatch["P_BAT_ch"], dispatch["P_BAT_dis"]
    )
    dispatch["thermal_storage_simultaneous_overlap"] = np.minimum(
        dispatch["H_TS_ch"], dispatch["H_TS_dis"]
    )
    dispatch["grid_import_export_overlap"] = np.minimum(
        dispatch["P_grid_buy"], dispatch["P_grid_sell"]
    )
    dispatch["flexible_load_overlap"] = np.minimum(
        dispatch["P_shift_in"], dispatch["P_shift_out"]
    )

    dispatch["electric_balance_error"] = (
        dispatch["P_grid_buy"]
        - dispatch["P_grid_sell"]
        + dispatch["P_PV"]
        + dispatch["P_WT"]
        + dispatch["P_CHP"]
        + dispatch["P_BAT_dis"]
        + dispatch["P_unmet_e"]
        - dispatch["P_BAT_ch"]
        - dispatch["P_HP"]
        - dispatch["P_EC"]
        - dispatch["electric_load_adjusted"]
    )
    dispatch["heat_balance_error"] = (
        dispatch["H_CHP"]
        + dispatch["H_GB"]
        + dispatch["H_HP"]
        + dispatch["H_TS_dis"]
        + dispatch["H_unmet"]
        - dispatch["H_TS_ch"]
        - dispatch["heat_load"]
    )
    dispatch["cooling_balance_error"] = (
        dispatch["C_EC_out"] + dispatch["C_unmet"] - dispatch["cooling_load"]
    )

    # Day-internal ramps for reporting.
    for col in ["P_grid_net", "P_CHP", "H_HP", "C_EC_out", "H_GB"]:
        ramp = np.zeros(T)
        for times in day_groups.values():
            values = dispatch.loc[times, col].values
            ramp[times] = values - np.roll(values, 1)
        dispatch[f"{col}_ramp"] = ramp

    # -----------------------------
    # Annual cost accounting
    # -----------------------------
    capacity_map = dict(zip(capacity_keys, capacity_values))
    annualized_capital_cost = sum(
        annualized_capital_cost_per_unit[key] * capacity_map[key]
        for key in capacity_keys
    )
    annual_fixed_om_cost = sum(
        annual_fixed_om_cost_per_unit[key] * capacity_map[key]
        for key in capacity_keys
    )
    annual_investment_cost = annualized_capital_cost + annual_fixed_om_cost
    annual_grid_purchase_cost = float(
        (dispatch["P_grid_buy"] * buy_price * dt * weights).sum()
    )
    annual_grid_sale_revenue = float(
        (dispatch["P_grid_sell"] * sell_price * dt * weights).sum()
    )
    annual_grid_export_wheeling_cost = float(
        (dispatch["P_grid_sell"] * grid_export_wheeling_charge * dt * weights).sum()
    )
    annual_gas_purchase_cost = float(
        ((dispatch["G_CHP"] + dispatch["G_GB"]) * profiles["gas_price"].values * dt * weights).sum()
    )
    annual_carbon_cost = float(
        (dispatch["co2_emission"] * profiles["carbon_price"].values * weights).sum()
    )
    annual_dr_cost = float(
        (
            0.5 * load_shifting_cost * (dispatch["P_shift_in"] + dispatch["P_shift_out"])
            + interruptible_load_cost * dispatch["P_interrupt"]
        )
        .mul(dt * weights)
        .sum()
    )
    annual_curtailment_cost = float(
        (renewable_curtailment_penalty * dispatch["renewable_curtailment"] * dt * weights).sum()
    )
    annual_unmet_cost = float(
        (
            electric_unserved_energy_penalty * dispatch["P_unmet_e"]
            + heat_unserved_energy_penalty * dispatch["H_unmet"]
            + cooling_unserved_energy_penalty * dispatch["C_unmet"]
        ).mul(dt * weights).sum()
    )
    annual_storage_degradation_cost = float(
        (
            0.5 * battery_degradation_cost * (dispatch["P_BAT_ch"] + dispatch["P_BAT_dis"])
            + 0.5 * thermal_storage_degradation_cost * (dispatch["H_TS_ch"] + dispatch["H_TS_dis"])
        )
        .mul(dt * weights)
        .sum()
    )
    annual_grid_demand_cost = float(grid_demand_charge * x[v("P_grid_peak")])
    annual_grid_export_capacity_cost = float(
        grid_export_capacity_charge * x[v("P_grid_export_peak")]
    )
    annual_grid_smoothing_cost = float(
        grid_smoothing_penalty
        * ((x[idx["D_GRID_up"]] + x[idx["D_GRID_down"]]) * weights).sum()
    )
    annual_device_smoothing_cost = 0.0
    for prefix in ["CHP", "HP", "EC", "GB"]:
        annual_device_smoothing_cost += float(
            device_smoothing_penalty
            * ((x[idx[f"D_{prefix}_up"]] + x[idx[f"D_{prefix}_down"]]) * weights).sum()
        )

    variable_operating_annual_cost = (
        annual_grid_purchase_cost
        - annual_grid_sale_revenue
        + annual_gas_purchase_cost
        + annual_grid_export_wheeling_cost
        + annual_grid_export_capacity_cost
        + annual_dr_cost
        + annual_curtailment_cost
        + annual_unmet_cost
        + annual_storage_degradation_cost
        + annual_grid_demand_cost
        + annual_grid_smoothing_cost
        + annual_device_smoothing_cost
    )
    private_annual_cost = (
        annualized_capital_cost + annual_fixed_om_cost + variable_operating_annual_cost
    )
    carbon_external_cost = annual_carbon_cost
    social_annual_cost = private_annual_cost + carbon_external_cost
    objective_annual_cost = private_annual_cost + (carbon_external_cost if internalize_carbon_price else 0.0)

    cost_df = pd.DataFrame(
        {
            "Cost item": [
                "Annualized capital cost",
                "Fixed O&M cost",
                "Grid purchase cost",
                "Grid sale revenue",
                "Grid export wheeling cost",
                "Grid export capacity cost",
                "Gas purchase cost",
                "Carbon external cost",
                "Demand response cost",
                "Renewable curtailment cost",
                "Unserved-energy cost",
                "Storage degradation cost",
                "Grid demand charge",
                "Grid fluctuation cost",
                "Device smoothing cost",
                "Variable operating annual cost",
                "Private total annual cost",
                "Social total annual cost",
                "Objective annual cost",
            ],
            "Value": [
                annualized_capital_cost,
                annual_fixed_om_cost,
                annual_grid_purchase_cost,
                annual_grid_sale_revenue,
                annual_grid_export_wheeling_cost,
                annual_grid_export_capacity_cost,
                annual_gas_purchase_cost,
                carbon_external_cost,
                annual_dr_cost,
                annual_curtailment_cost,
                annual_unmet_cost,
                annual_storage_degradation_cost,
                annual_grid_demand_cost,
                annual_grid_smoothing_cost,
                annual_device_smoothing_cost,
                variable_operating_annual_cost,
                private_annual_cost,
                social_annual_cost,
                objective_annual_cost,
            ],
            "Unit": ["EUR/year"] * 19,
        }
    )

    annual_co2 = float(dispatch["co2_emission_annual"].sum())
    daily_average_co2 = annual_co2 / annual_days
    co2_cap = float(scenario.get("co2_cap", np.nan))
    co2_slack = co2_cap - daily_average_co2 if np.isfinite(co2_cap) else np.nan
    carbon_df = pd.DataFrame(
        {
            "Item": [
                "Daily average CO2 emissions",
                "Annual CO2 emissions",
                "CO2 cap",
                "CO2 cap slack",
            ],
            "Value": [daily_average_co2, annual_co2, co2_cap, co2_slack],
            "Unit": ["tCO2/day", "tCO2/year", "tCO2/day", "tCO2/day"],
        }
    )

    annual_renewable_available = float(
        (dispatch["renewable_available"] * dt * weights).sum()
    )
    annual_renewable_used = float(
        (dispatch["renewable_used"] * dt * weights).sum()
    )
    annual_renewable_curtailment = float(
        (dispatch["renewable_curtailment"] * dt * weights).sum()
    )
    renewable_utilization = (
        100.0 * annual_renewable_used / annual_renewable_available
        if annual_renewable_available > 1e-9
        else 0.0
    )
    renewable_curtailment_rate = (
        100.0 * annual_renewable_curtailment / annual_renewable_available
        if annual_renewable_available > 1e-9
        else 0.0
    )
    annual_electric_unmet = float((dispatch["P_unmet_e"] * dt * weights).sum())
    annual_heat_unmet = float((dispatch["H_unmet"] * dt * weights).sum())
    annual_cooling_unmet = float((dispatch["C_unmet"] * dt * weights).sum())
    annual_unmet = annual_electric_unmet + annual_heat_unmet + annual_cooling_unmet
    annual_electric_demand = float((dispatch["electric_load"] * dt * weights).sum())
    annual_heat_demand = float((dispatch["heat_load"] * dt * weights).sum())
    annual_cooling_demand = float((dispatch["cooling_load"] * dt * weights).sum())
    annual_total_demand = annual_electric_demand + annual_heat_demand + annual_cooling_demand
    electric_service_rate = 100.0 * (1.0 - annual_electric_unmet / annual_electric_demand) if annual_electric_demand > 1e-9 else 100.0
    heat_service_rate = 100.0 * (1.0 - annual_heat_unmet / annual_heat_demand) if annual_heat_demand > 1e-9 else 100.0
    cooling_service_rate = 100.0 * (1.0 - annual_cooling_unmet / annual_cooling_demand) if annual_cooling_demand > 1e-9 else 100.0
    total_service_rate = 100.0 * (1.0 - annual_unmet / annual_total_demand) if annual_total_demand > 1e-9 else 100.0
    unserved_energy_ratio = 100.0 * annual_unmet / annual_total_demand if annual_total_demand > 1e-9 else 0.0
    shortage_mask = (dispatch["P_unmet_e"] + dispatch["H_unmet"] + dispatch["C_unmet"]) > 1e-6
    expected_shortage_duration = float((shortage_mask.astype(float) * dt * weights).sum())
    annual_grid_import = float((dispatch["P_grid_buy"] * dt * weights).sum())
    annual_grid_export = float((dispatch["P_grid_sell"] * dt * weights).sum())
    # Grid export is constrained by contemporaneous PV+WT generation, so it can
    # be consistently attributed to renewable export in this aggregated model.
    annual_renewable_export = annual_grid_export
    annual_renewable_local = max(0.0, annual_renewable_used - annual_renewable_export)
    renewable_local_consumption_rate = (
        100.0 * annual_renewable_local / annual_renewable_used
        if annual_renewable_used > 1e-9 else 0.0
    )
    renewable_export_rate = (
        100.0 * annual_renewable_export / annual_renewable_available
        if annual_renewable_available > 1e-9 else 0.0
    )
    renewable_local_absorption_rate = (
        100.0 * annual_renewable_local / annual_renewable_available
        if annual_renewable_available > 1e-9 else 0.0
    )
    annual_battery_throughput = float(
        (0.5 * (dispatch["P_BAT_ch"] + dispatch["P_BAT_dis"]) * dt * weights).sum()
    )
    annual_ts_throughput = float(
        (0.5 * (dispatch["H_TS_ch"] + dispatch["H_TS_dis"]) * dt * weights).sum()
    )
    annual_flexible_shift = float(
        (0.5 * (dispatch["P_shift_in"] + dispatch["P_shift_out"]) * dt * weights).sum()
    )
    annual_interruptible = float((dispatch["P_interrupt"] * dt * weights).sum())
    annual_battery_overlap = float(
        (dispatch["battery_simultaneous_overlap"] * dt * weights).sum()
    )
    annual_ts_overlap = float(
        (dispatch["thermal_storage_simultaneous_overlap"] * dt * weights).sum()
    )
    annual_grid_overlap = float(
        (dispatch["grid_import_export_overlap"] * dt * weights).sum()
    )
    annual_flexible_overlap = float(
        (dispatch["flexible_load_overlap"] * dt * weights).sum()
    )

    grid_net = dispatch["P_grid_net"].values
    grid_weight = dispatch["day_weight"].values
    max_grid_ramp = float(np.max(np.abs(dispatch["P_grid_net_ramp"].values)))
    peak_import = float(dispatch["P_grid_buy"].max())
    peak_export = float(dispatch["P_grid_sell"].max())
    peak_valley = float(np.max(grid_net) - np.min(grid_net))
    grid_std = _weighted_std(grid_net, grid_weight)

    battery_energy_capacity = capacity_map["C_BAT_E"]
    ts_energy_capacity = capacity_map["C_TS_E"]
    battery_equivalent_cycles = (
        annual_battery_throughput / battery_energy_capacity
        if battery_energy_capacity > 1e-9
        else 0.0
    )
    ts_equivalent_cycles = (
        annual_ts_throughput / ts_energy_capacity
        if ts_energy_capacity > 1e-9
        else 0.0
    )

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Annualized capital cost",
                "Fixed O&M cost",
                "Variable operating annual cost",
                "Private total annual cost",
                "Carbon external cost",
                "Social total annual cost",
                "Annual CO2 emissions",
                "Annual renewable available",
                "Annual renewable used",
                "Renewable utilization rate (including export)",
                "Renewable curtailment rate",
                "Annual renewable curtailment",
                "Annual renewable local absorption",
                "Annual renewable export",
                "Renewable local consumption rate",
                "Renewable local absorption rate",
                "Renewable export rate",
                "Annual unmet load",
                "Annual electric unmet load",
                "Annual heat unmet load",
                "Annual cooling unmet load",
                "Electric service rate",
                "Heat service rate",
                "Cooling service rate",
                "Total multi-energy service rate",
                "Unserved energy ratio",
                "Expected shortage duration",
                "Annual grid import",
                "Annual grid export",
                "Peak grid import",
                "Peak grid export",
                "Grid peak-valley difference",
                "Grid weighted standard deviation",
                "Maximum grid ramp",
                "Annual battery throughput",
                "Battery equivalent cycles",
                "Annual thermal-storage throughput",
                "Thermal-storage equivalent cycles",
                "Annual flexible-load shifting",
                "Annual interruptible load",
                "Battery simultaneous overlap",
                "Thermal-storage simultaneous overlap",
                "Grid import-export overlap",
                "Flexible-load in-out overlap",
            ],
            "Value": [
                annualized_capital_cost,
                annual_fixed_om_cost,
                variable_operating_annual_cost,
                private_annual_cost,
                carbon_external_cost,
                social_annual_cost,
                annual_co2,
                annual_renewable_available,
                annual_renewable_used,
                renewable_utilization,
                renewable_curtailment_rate,
                annual_renewable_curtailment,
                annual_renewable_local,
                annual_renewable_export,
                renewable_local_consumption_rate,
                renewable_local_absorption_rate,
                renewable_export_rate,
                annual_unmet,
                annual_electric_unmet,
                annual_heat_unmet,
                annual_cooling_unmet,
                electric_service_rate,
                heat_service_rate,
                cooling_service_rate,
                total_service_rate,
                unserved_energy_ratio,
                expected_shortage_duration,
                annual_grid_import,
                annual_grid_export,
                peak_import,
                peak_export,
                peak_valley,
                grid_std,
                max_grid_ramp,
                annual_battery_throughput,
                battery_equivalent_cycles,
                annual_ts_throughput,
                ts_equivalent_cycles,
                annual_flexible_shift,
                annual_interruptible,
                annual_battery_overlap,
                annual_ts_overlap,
                annual_grid_overlap,
                annual_flexible_overlap,
            ],
            "Unit": [
                "EUR/year", "EUR/year", "EUR/year", "EUR/year",
                "EUR/year", "EUR/year", "tCO2/year",
                "MWh/year", "MWh/year", "%", "%",
                "MWh/year", "MWh/year", "MWh/year", "%", "%", "%",
                "MWh/year", "MWh/year", "MWh/year", "MWh/year",
                "%", "%", "%", "%", "%", "h/year",
                "MWh/year", "MWh/year", "MW", "MW", "MW", "MW",
                "MW/step", "MWh/year", "cycles/year", "MWh/year",
                "cycles/year", "MWh/year", "MWh/year", "MWh/year",
                "MWh/year", "MWh/year", "MWh/year",
            ],
        }
    )

    diagnostics_df = pd.DataFrame(
        {
            "Diagnostic": [
                "Maximum electric balance error",
                "Maximum heat balance error",
                "Maximum cooling balance error",
                "Number of capacity upper bounds >=95%",
                "Strict storage exclusivity enabled",
                "MILP used",
                "Carbon cap binding",
                "Annual export share of renewable available [%]",
                "Storage rule-constrained mode",
                "Grid import-export overlap [MWh/year]",
                "Flexible-load in-out overlap [MWh/year]",
                "Renewable export attribution enforced",
                "Unmet load hard-fixed to zero",
                "Annual EENS limit enabled",
                "Critical-load protection enabled",
            ],
            "Value": [
                float(dispatch["electric_balance_error"].abs().max()),
                float(dispatch["heat_balance_error"].abs().max()),
                float(dispatch["cooling_balance_error"].abs().max()),
                int(capacity_df["Upper bound binding"].sum()),
                int(strict_storage_exclusivity),
                int(np.any(integrality_array != 0)),
                int(enable_carbon_constraint and abs(co2_slack) <= max(1e-5, 1e-4 * max(abs(co2_cap), 1.0))),
                100.0 * annual_grid_export / annual_renewable_available if annual_renewable_available > 1e-9 else 0.0,
                int(storage_dispatch_mode == "rule"),
                annual_grid_overlap,
                annual_flexible_overlap,
                int(enable_grid_export),
                int(force_zero_unmet),
                int(max_annual_unserved_energy_ratio is not None),
                1,
            ],
        }
    )

    return {
        "success": True,
        "message": "Optimization solved successfully.",
        "solver_status": int(result.status),
        "capacity": capacity_df,
        "dispatch": dispatch,
        "cost": cost_df,
        "carbon": carbon_df,
        "metrics": metrics_df,
        "diagnostics": diagnostics_df,
        "objective": float(result.fun),
        "annual_days": annual_days,
        "dt": dt,
    }


if __name__ == "__main__":
    from data_profiles_v17_3_1_tongjiang import generate_tongjiang_characteristic_profiles_v17_3_1
    from scenarios_v17_3_1_county import get_planning_scenarios_v17_3_1

    profiles = generate_tongjiang_characteristic_profiles_v17_3_1(n_steps_per_hour=2)
    scenario = get_planning_scenarios_v17_3_1()["S4"]
    result = solve_integrated_energy_system_v17_3_1(profiles, scenario)
    print(result["message"])
    if result["success"]:
        print(result["capacity"].round(3))
        print(result["cost"].round(3))
        print(result["metrics"].round(3))
