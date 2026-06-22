from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd

from data_profiles_v17_3_1_tongjiang import generate_tongjiang_characteristic_profiles_v17_3_1
from model_core_v17_3_1_county import (
    get_annual_days,
    infer_dt,
    solve_integrated_energy_system_v17_3_1,
)
from scenarios_v17_3_1_county import (
    get_operation_scenarios_v17_3_1,
    get_planning_scenarios_v17_3_1,
)


RESULT_ROOT = Path("results_v17_3_1_county_tongjiang")
CNY_PER_EUR = 7.8
N_STEPS_PER_HOUR = int(os.environ.get("V17_STEPS_PER_HOUR", "4"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_table_value(
    df: pd.DataFrame,
    key_column: str,
    key: str,
    value_column: str = "Value",
) -> float:
    row = df[df[key_column] == key]
    if row.empty:
        return 0.0
    return float(row[value_column].iloc[0])


def capacities_to_fixed_dict(capacity_df: pd.DataFrame) -> dict[str, float]:
    return {
        str(row["Model key"]): float(row["Capacity"])
        for _, row in capacity_df.iterrows()
    }


def save_result(result: dict, scenario_dir: Path) -> None:
    ensure_dir(scenario_dir)
    result["capacity"].to_csv(scenario_dir / "capacity_results.csv", index=False, encoding="utf-8-sig")
    result["dispatch"].to_csv(scenario_dir / "dispatch_results.csv", index=False, encoding="utf-8-sig")
    result["cost"].to_csv(scenario_dir / "cost_results.csv", index=False, encoding="utf-8-sig")
    result["carbon"].to_csv(scenario_dir / "carbon_results.csv", index=False, encoding="utf-8-sig")
    result["metrics"].to_csv(scenario_dir / "metrics_results.csv", index=False, encoding="utf-8-sig")
    result["diagnostics"].to_csv(scenario_dir / "diagnostics_results.csv", index=False, encoding="utf-8-sig")


def summarize_result(scenario_key: str, scenario: dict, result: dict, group: str) -> dict:
    capacity = result["capacity"]
    metrics = result["metrics"]
    carbon = result["carbon"]
    diagnostics = result["diagnostics"]

    def cap(name: str) -> float:
        row = capacity[capacity["Technology"] == name]
        return float(row["Capacity"].iloc[0]) if not row.empty else 0.0

    def metric(name: str) -> float:
        return get_table_value(metrics, "Metric", name)

    def carbon_value(name: str) -> float:
        return get_table_value(carbon, "Item", name)

    return {
        "Group": group,
        "Scenario": scenario_key,
        "Name": scenario.get("name", scenario_key),
        "Success": True,
        "Message": result["message"],
        "Annualized capital cost [EUR/year]": metric("Annualized capital cost"),
        "Annualized capital cost [million CNY/year]": metric("Annualized capital cost") * CNY_PER_EUR / 1e6,
        "Fixed O&M cost [EUR/year]": metric("Fixed O&M cost"),
        "Fixed O&M cost [million CNY/year]": metric("Fixed O&M cost") * CNY_PER_EUR / 1e6,
        "Variable operating annual cost [EUR/year]": metric("Variable operating annual cost"),
        "Variable operating annual cost [million CNY/year]": metric("Variable operating annual cost") * CNY_PER_EUR / 1e6,
        "Private total annual cost [EUR/year]": metric("Private total annual cost"),
        "Private total annual cost [million CNY/year]": metric("Private total annual cost") * CNY_PER_EUR / 1e6,
        "Carbon external cost [EUR/year]": metric("Carbon external cost"),
        "Carbon external cost [million CNY/year]": metric("Carbon external cost") * CNY_PER_EUR / 1e6,
        "Social total annual cost [EUR/year]": metric("Social total annual cost"),
        "Social total annual cost [million CNY/year]": metric("Social total annual cost") * CNY_PER_EUR / 1e6,
        "Annual CO2 emissions [tCO2/year]": metric("Annual CO2 emissions"),
        "Daily average CO2 emissions [tCO2/day]": carbon_value("Daily average CO2 emissions"),
        "CO2 cap [tCO2/day]": carbon_value("CO2 cap"),
        "CO2 cap slack [tCO2/day]": carbon_value("CO2 cap slack"),
        "Renewable utilization rate incl. export [%]": metric("Renewable utilization rate (including export)"),
        "Renewable curtailment rate [%]": metric("Renewable curtailment rate"),
        "Renewable local consumption rate [%]": metric("Renewable local consumption rate"),
        "Renewable local absorption rate [%]": metric("Renewable local absorption rate"),
        "Renewable export rate [%]": metric("Renewable export rate"),
        "Annual renewable available [MWh/year]": metric("Annual renewable available"),
        "Annual renewable local absorption [MWh/year]": metric("Annual renewable local absorption"),
        "Annual renewable export [MWh/year]": metric("Annual renewable export"),
        "Annual renewable curtailment [MWh/year]": metric("Annual renewable curtailment"),
        "Annual unmet load [MWh/year]": metric("Annual unmet load"),
        "Electric service rate [%]": metric("Electric service rate"),
        "Heat service rate [%]": metric("Heat service rate"),
        "Cooling service rate [%]": metric("Cooling service rate"),
        "Total multi-energy service rate [%]": metric("Total multi-energy service rate"),
        "Unserved energy ratio [%]": metric("Unserved energy ratio"),
        "Expected shortage duration [h/year]": metric("Expected shortage duration"),
        "Annual grid import [MWh/year]": metric("Annual grid import"),
        "Annual grid export [MWh/year]": metric("Annual grid export"),
        "Peak grid import [MW]": metric("Peak grid import"),
        "Peak grid export [MW]": metric("Peak grid export"),
        "Grid peak-valley difference [MW]": metric("Grid peak-valley difference"),
        "Grid weighted standard deviation [MW]": metric("Grid weighted standard deviation"),
        "Maximum grid ramp [MW/step]": metric("Maximum grid ramp"),
        "Annual battery throughput [MWh/year]": metric("Annual battery throughput"),
        "Battery equivalent cycles [cycles/year]": metric("Battery equivalent cycles"),
        "Annual thermal-storage throughput [MWh/year]": metric("Annual thermal-storage throughput"),
        "Thermal-storage equivalent cycles [cycles/year]": metric("Thermal-storage equivalent cycles"),
        "Annual flexible-load shifting [MWh/year]": metric("Annual flexible-load shifting"),
        "Battery simultaneous overlap [MWh/year]": metric("Battery simultaneous overlap"),
        "Thermal-storage simultaneous overlap [MWh/year]": metric("Thermal-storage simultaneous overlap"),
        "Grid import-export overlap [MWh/year]": metric("Grid import-export overlap"),
        "Flexible-load in-out overlap [MWh/year]": metric("Flexible-load in-out overlap"),
        "PV capacity [MW]": cap("PV"),
        "WT capacity [MW]": cap("WT"),
        "CHP capacity [MW]": cap("CHP"),
        "Heat pump capacity [MW]": cap("Heat pump"),
        "Electric chiller capacity [MW]": cap("Electric chiller"),
        "Gas boiler capacity [MW]": cap("Gas boiler"),
        "Battery energy capacity [MWh]": cap("Battery energy"),
        "Battery power capacity [MW]": cap("Battery power"),
        "Thermal storage energy capacity [MWh]": cap("Thermal storage energy"),
        "Thermal storage power capacity [MW]": cap("Thermal storage power"),
        "Capacity upper bounds >=95% [count]": get_table_value(
            diagnostics, "Diagnostic", "Number of capacity upper bounds >=95%"
        ),
        "Carbon cap binding": get_table_value(diagnostics, "Diagnostic", "Carbon cap binding"),
        "Annual export share of renewable available [%]": get_table_value(
            diagnostics, "Diagnostic", "Annual export share of renewable available [%]"
        ),
        "Max electric balance error": get_table_value(
            diagnostics, "Diagnostic", "Maximum electric balance error"
        ),
        "Max heat balance error": get_table_value(
            diagnostics, "Diagnostic", "Maximum heat balance error"
        ),
        "Max cooling balance error": get_table_value(
            diagnostics, "Diagnostic", "Maximum cooling balance error"
        ),
    }


def _cyclic_rule_profiles(
    profiles: pd.DataFrame,
    power_capacity: float,
    energy_capacity: float,
    initial_soc: float,
    eta_ch: float,
    eta_dis: float,
    self_discharge_per_hour: float,
    charge_fraction: float,
    charge_windows: list[tuple[float, float]],
    discharge_windows: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic rule profiles with approximate daily SOC closure."""

    T = len(profiles)
    dt = float(profiles["dt"].iloc[0])
    charge = np.zeros(T)
    discharge = np.zeros(T)
    if power_capacity <= 1e-9 or energy_capacity <= 1e-9:
        return charge, discharge

    for _, sub in profiles.groupby("day_id", sort=True):
        indices = sub.index.to_numpy()
        hour = sub["hour"].to_numpy(float)
        ch_shape = np.array([
            1.0 if any(a <= h < b for a, b in charge_windows) else 0.0
            for h in hour
        ])
        dis_shape = np.array([
            1.0 if any(a <= h < b for a, b in discharge_windows) else 0.0
            for h in hour
        ])
        if ch_shape.sum() == 0 or dis_shape.sum() == 0:
            continue

        charge_power = charge_fraction * power_capacity
        ch = charge_power * ch_shape
        retention = max(0.0, 1.0 - self_discharge_per_hour * dt)
        n = len(indices)
        discount = retention ** np.arange(n - 1, -1, -1)
        final_without_discharge = (
            retention**n * initial_soc * energy_capacity
            + np.sum(discount * eta_ch * ch * dt)
        )
        target_final = initial_soc * energy_capacity
        denominator = np.sum(discount * dis_shape * dt / eta_dis)
        dis_power = max(0.0, (final_without_discharge - target_final) / max(denominator, 1e-9))

        if dis_power > 0.85 * power_capacity:
            scale = (0.85 * power_capacity) / dis_power
            ch *= scale
            final_without_discharge = (
                retention**n * initial_soc * energy_capacity
                + np.sum(discount * eta_ch * ch * dt)
            )
            dis_power = max(0.0, (final_without_discharge - target_final) / max(denominator, 1e-9))

        charge[indices] = ch
        discharge[indices] = dis_power * dis_shape

    return charge, discharge


def build_rule_dispatch_profiles(
    profiles: pd.DataFrame,
    fixed_capacities: dict[str, float],
) -> dict[str, np.ndarray]:
    bat_ch, bat_dis = _cyclic_rule_profiles(
        profiles,
        power_capacity=fixed_capacities.get("C_BAT_P", 0.0),
        energy_capacity=fixed_capacities.get("C_BAT_E", 0.0),
        initial_soc=0.50,
        eta_ch=0.95,
        eta_dis=0.95,
        self_discharge_per_hour=0.0002,
        charge_fraction=0.12,
        charge_windows=[(0.0, 5.5), (11.0, 15.0)],
        discharge_windows=[(7.0, 9.5), (17.0, 21.5)],
    )
    ts_ch, ts_dis = _cyclic_rule_profiles(
        profiles,
        power_capacity=fixed_capacities.get("C_TS_P", 0.0),
        energy_capacity=fixed_capacities.get("C_TS_E", 0.0),
        initial_soc=0.50,
        eta_ch=0.95,
        eta_dis=0.95,
        self_discharge_per_hour=0.0020,
        charge_fraction=0.10,
        charge_windows=[(0.0, 5.5), (11.0, 15.5)],
        discharge_windows=[(6.0, 9.5), (17.0, 22.0)],
    )
    return {
        "fixed_battery_charge_profile": bat_ch,
        "fixed_battery_discharge_profile": bat_dis,
        "fixed_thermal_charge_profile": ts_ch,
        "fixed_thermal_discharge_profile": ts_dis,
    }


def build_comparison(
    summary: pd.DataFrame,
    baseline_key: str,
    optimized_key: str,
) -> pd.DataFrame:
    a = summary[summary["Scenario"] == baseline_key].iloc[0]
    b = summary[summary["Scenario"] == optimized_key].iloc[0]
    definitions = [
        ("年可变运行成本", "Variable operating annual cost [million CNY/year]", "lower"),
        ("年私人成本总额", "Private total annual cost [million CNY/year]", "lower"),
        ("年社会综合成本", "Social total annual cost [million CNY/year]", "lower"),
        ("年碳排放", "Annual CO2 emissions [tCO2/year]", "lower"),
        ("主网最大购电功率", "Peak grid import [MW]", "lower"),
        ("主网峰谷差", "Grid peak-valley difference [MW]", "lower"),
        ("主网功率标准差", "Grid weighted standard deviation [MW]", "lower"),
        ("年弃风弃光量", "Annual renewable curtailment [MWh/year]", "lower"),
        ("弃风弃光率", "Renewable curtailment rate [%]", "lower"),
        ("新能源本地吸纳率", "Renewable local absorption rate [%]", "higher"),
        ("新能源外送率", "Renewable export rate [%]", "lower"),
        ("年购电量", "Annual grid import [MWh/year]", "lower"),
    ]
    rows = []
    for label, column, direction in definitions:
        before = float(a[column])
        after = float(b[column])
        improvement = before - after if direction == "lower" else after - before
        rate = 100.0 * improvement / abs(before) if abs(before) > 1e-9 else np.nan
        rows.append(
            {
                "指标": label,
                f"{baseline_key}基准": before,
                f"{optimized_key}优化": after,
                "改善量": improvement,
                "改善率/%": rate,
                "优选方向": direction,
            }
        )
    return pd.DataFrame(rows)


def build_tongjiang_feature_summary(
    profiles: pd.DataFrame,
    r0_dispatch: pd.DataFrame,
    r4_dispatch: pd.DataFrame,
) -> pd.DataFrame:
    key_days = ["spring_irrigation", "spring_pv_high", "autumn_processing", "winter_heating"]
    rows = []
    for day_type in key_days:
        meta = profiles[profiles["day_type"] == day_type].iloc[0]
        for scenario, dispatch in [("R0", r0_dispatch), ("R4", r4_dispatch)]:
            sub = dispatch[dispatch["day_type"] == day_type]
            dt = float(sub["dt"].iloc[0])
            rows.append(
                {
                    "Scenario": scenario,
                    "Day type": day_type,
                    "Day name": meta["day_name"],
                    "Tongjiang feature": meta["tongjiang_feature"],
                    "Peak import [MW]": float(sub["P_grid_buy"].max()),
                    "Peak export [MW]": float(sub["P_grid_sell"].max()),
                    "Grid exchange range [MW]": float(sub["P_grid_net"].max() - sub["P_grid_net"].min()),
                    "Renewable curtailment [MWh/day]": float((sub["renewable_curtailment"] * dt).sum()),
                    "Battery throughput [MWh/day]": float((0.5 * (sub["P_BAT_ch"] + sub["P_BAT_dis"]) * dt).sum()),
                    "Thermal storage throughput [MWh/day]": float((0.5 * (sub["H_TS_ch"] + sub["H_TS_dis"]) * dt).sum()),
                    "Flexible load shifting [MWh/day]": float((0.5 * (sub["P_shift_in"] + sub["P_shift_out"]) * dt).sum()),
                    "Maximum grid ramp [MW/step]": float(sub["P_grid_net_ramp"].abs().max()),
                }
            )
    return pd.DataFrame(rows)


def run_v17_3_1() -> None:
    ensure_dir(RESULT_ROOT)
    profiles = generate_tongjiang_characteristic_profiles_v17_3_1(
        n_steps_per_hour=N_STEPS_PER_HOUR,
        seed=42,
    )
    profiles.to_csv(
        RESULT_ROOT / "input_profiles_v17_3_1_tongjiang_characteristic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Starting V17.3.1 county-scale model with Tongjiang-characteristic profiles")
    print("Important: the profiles are feature-based synthetic data, not measured feeder data.")
    print(f"Time steps: {len(profiles)}")
    print(f"dt: {infer_dt(profiles)} h")
    print(f"Annual representative-day weight: {get_annual_days(profiles)} days")
    print()

    planning_scenarios = get_planning_scenarios_v17_3_1()
    planning_rows: list[dict] = []
    planning_results: dict[str, dict] = {}

    s0 = solve_integrated_energy_system_v17_3_1(profiles, planning_scenarios["S0"])
    if not s0["success"]:
        raise RuntimeError(f"S0 failed: {s0['message']}")
    planning_results["S0"] = s0
    save_result(s0, RESULT_ROOT / "planning" / "S0" / "data")
    planning_rows.append(summarize_result("S0", planning_scenarios["S0"], s0, "Planning"))

    baseline_daily_co2 = get_table_value(s0["carbon"], "Item", "Daily average CO2 emissions")
    cap_ratio = float(planning_scenarios["S1"].get("co2_cap_ratio_to_s0", 0.85))
    common_cap = baseline_daily_co2 * cap_ratio

    for key in ["S1", "S2", "S3", "S4"]:
        scenario = dict(planning_scenarios[key])
        scenario["co2_cap"] = common_cap
        print(f"Solving {key}: {scenario['name']}")
        result = solve_integrated_energy_system_v17_3_1(profiles, scenario)
        if not result["success"]:
            raise RuntimeError(f"{key} failed: {result['message']}")
        planning_results[key] = result
        save_result(result, RESULT_ROOT / "planning" / key / "data")
        planning_rows.append(summarize_result(key, scenario, result, "Planning"))

    planning_summary = pd.DataFrame(planning_rows)
    planning_summary.to_csv(
        RESULT_ROOT / "planning_summary_v17_3_1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fixed_capacities = capacities_to_fixed_dict(planning_results["S4"]["capacity"])
    operation_scenarios = get_operation_scenarios_v17_3_1(fixed_capacities)
    rule_profiles = build_rule_dispatch_profiles(profiles, fixed_capacities)
    for key in ["R0", "R2"]:
        operation_scenarios[key].update(rule_profiles)

    operation_rows: list[dict] = []
    operation_results: dict[str, dict] = {}
    for key in ["R0", "R1", "R2", "R3"]:
        scenario = dict(operation_scenarios[key])
        print(f"Solving {key}: {scenario['name']}")
        result = solve_integrated_energy_system_v17_3_1(profiles, scenario)
        if not result["success"]:
            raise RuntimeError(f"{key} failed: {result['message']}")
        operation_results[key] = result
        save_result(result, RESULT_ROOT / "operation" / key / "data")
        operation_rows.append(summarize_result(key, scenario, result, "Operation"))

    # V17.3.1: use R3's natural economic-dispatch emissions as the reference.
    # The formal R4 cap is fixed at 97% of R3 emissions, so R3 and R4 have a
    # clear and reproducible economic-versus-low-carbon comparison.
    r3_daily_co2 = get_table_value(
        operation_results["R3"]["carbon"], "Item", "Daily average CO2 emissions"
    )
    target_ratio = float(operation_scenarios["R4"].get("r4_cap_ratio_to_r3", 0.97))
    # Default 15-minute runs normally use 97%. Coarser debugging resolutions
    # may make this exact target infeasible, so progressively relax only as
    # much as necessary while keeping the cap below R3 emissions.
    candidate_ratios = []
    for ratio in [target_ratio, 0.975, 0.98, 0.985, 0.99]:
        if ratio >= target_ratio - 1e-12 and ratio < 1.0 and ratio not in candidate_ratios:
            candidate_ratios.append(ratio)

    r4 = None
    r4_scenario = None
    selected_ratio = None
    last_message = ""
    for ratio in candidate_ratios:
        candidate = dict(operation_scenarios["R4"])
        candidate["co2_cap"] = r3_daily_co2 * ratio
        print(
            f"Trying R4 hard carbon cap = {ratio:.1%} of R3 natural emissions "
            f"({candidate['co2_cap']:.6f} tCO2/day)"
        )
        trial = solve_integrated_energy_system_v17_3_1(profiles, candidate)
        if trial["success"]:
            r4 = trial
            r4_scenario = candidate
            selected_ratio = ratio
            break
        last_message = trial.get("message", "unknown solver failure")

    if r4 is None or r4_scenario is None or selected_ratio is None:
        raise RuntimeError(
            "R4 failed for all carbon caps from 97% to 99% of R3 emissions. "
            f"Last solver message: {last_message}"
        )
    if selected_ratio > target_ratio + 1e-12:
        print(
            f"Warning: exact {target_ratio:.1%} cap was infeasible at the current time resolution; "
            f"the tightest feasible cap was {selected_ratio:.1%}."
        )
    operation_results["R4"] = r4
    save_result(r4, RESULT_ROOT / "operation" / "R4" / "data")
    operation_rows.append(summarize_result("R4", r4_scenario, r4, "Operation"))

    operation_summary = pd.DataFrame(operation_rows)
    operation_summary.to_csv(
        RESULT_ROOT / "operation_summary_v17_3_1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    build_comparison(operation_summary, "R0", "R4").to_csv(
        RESULT_ROOT / "effectiveness_R0_vs_R4_v17_3_1.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_comparison(operation_summary, "R3", "R4").to_csv(
        RESULT_ROOT / "low_carbon_effect_R3_vs_R4_v17_3_1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tongjiang_summary = build_tongjiang_feature_summary(
        profiles,
        operation_results["R0"]["dispatch"],
        operation_results["R4"]["dispatch"],
    )
    tongjiang_summary.to_csv(
        RESULT_ROOT / "tongjiang_characteristic_day_summary_v17_3_1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nV17.3.1 main run completed.")
    print(f"Results: {RESULT_ROOT.resolve()}")
    print("Run sensitivity_run_v17_3_1_county.py for carbon/export sensitivity.")
    print("The Tongjiang module is characteristic screening, not a real feeder voltage model.")



if __name__ == "__main__":
    run_v17_3_1()
