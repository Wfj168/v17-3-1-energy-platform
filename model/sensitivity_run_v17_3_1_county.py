from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_profiles_v17_3_1_tongjiang import generate_tongjiang_characteristic_profiles_v17_3_1
from model_core_v17_3_1_county import solve_integrated_energy_system_v17_3_1
from paper_run_v17_3_1_county import (
    N_STEPS_PER_HOUR,
    RESULT_ROOT,
    capacities_to_fixed_dict,
    save_result,
    summarize_result,
)
from scenarios_v17_3_1_county import get_operation_scenarios_v17_3_1


def run_sensitivity_v17_3_1() -> None:
    planning_capacity_path = RESULT_ROOT / "planning" / "S4" / "data" / "capacity_results.csv"
    operation_summary_path = RESULT_ROOT / "operation_summary_v17_3_1.csv"
    if not planning_capacity_path.exists() or not operation_summary_path.exists():
        raise FileNotFoundError(
            "请先运行 paper_run_v17_3_1_county.py，生成S4容量和R3运行结果。"
        )

    profiles = generate_tongjiang_characteristic_profiles_v17_3_1(
        n_steps_per_hour=N_STEPS_PER_HOUR,
        seed=42,
    )
    fixed_capacities = capacities_to_fixed_dict(pd.read_csv(planning_capacity_path))
    operation_summary = pd.read_csv(operation_summary_path)
    r3_daily_co2 = float(
        operation_summary.loc[
            operation_summary["Scenario"] == "R3",
            "Daily average CO2 emissions [tCO2/day]",
        ].iloc[0]
    )
    operation_scenarios = get_operation_scenarios_v17_3_1(fixed_capacities)

    # Sensitivity calculations use the continuous relaxation.  Positive
    # degradation and export charges plus overlap diagnostics keep the result
    # physically interpretable while avoiding repeated large MILPs.
    carbon_rows = []
    for ratio in [1.00, 0.99, 0.98, 0.97]:
        key = f"C{int(round(ratio * 100))}"
        scenario = dict(operation_scenarios["R4"])
        scenario.update(
            {
                "name": f"{key} 碳上限为R3的{ratio:.0%}",
                "co2_cap": r3_daily_co2 * ratio,
                "strict_storage_exclusivity": False,
                "strict_grid_exchange_exclusivity": False,
            }
        )
        result = solve_integrated_energy_system_v17_3_1(profiles, scenario)
        if result["success"]:
            save_result(result, RESULT_ROOT / "sensitivity" / "carbon" / key / "data")
            carbon_rows.append(summarize_result(key, scenario, result, "Carbon sensitivity"))
        else:
            carbon_rows.append({"Scenario": key, "Success": False, "Message": result["message"]})
    pd.DataFrame(carbon_rows).to_csv(
        RESULT_ROOT / "carbon_cap_sensitivity_v17_3_1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    export_rows = []
    for export_capacity in [0.0, 2.5, 5.0, 10.0]:
        key = f"E{str(export_capacity).replace('.', '_')}"
        scenario = dict(operation_scenarios["R3"])
        scenario.update(
            {
                "name": f"外送上限{export_capacity:g} MW",
                "grid_export_capacity": export_capacity,
                "enable_grid_export": export_capacity > 1e-9,
                "strict_storage_exclusivity": False,
                "strict_grid_exchange_exclusivity": False,
            }
        )
        result = solve_integrated_energy_system_v17_3_1(profiles, scenario)
        if result["success"]:
            save_result(result, RESULT_ROOT / "sensitivity" / "export" / key / "data")
            row = summarize_result(key, scenario, result, "Export sensitivity")
            row["Export capacity limit [MW]"] = export_capacity
            export_rows.append(row)
        else:
            export_rows.append(
                {
                    "Scenario": key,
                    "Success": False,
                    "Message": result["message"],
                    "Export capacity limit [MW]": export_capacity,
                }
            )
    pd.DataFrame(export_rows).to_csv(
        RESULT_ROOT / "export_capacity_sensitivity_v17_3_1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("V17.3.1 sensitivity analysis completed.")
    print(RESULT_ROOT.resolve())


if __name__ == "__main__":
    run_sensitivity_v17_3_1()
