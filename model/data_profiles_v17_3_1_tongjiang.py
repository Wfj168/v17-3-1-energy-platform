from __future__ import annotations

import numpy as np
import pandas as pd


def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _smooth_noise(
    rng: np.random.Generator,
    n: int,
    scale: float = 1.0,
    window: int = 7,
) -> np.ndarray:
    raw = rng.normal(0.0, scale, n)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(raw, kernel, mode="same")


def _tou_buy_price(hour: np.ndarray, day_type: str) -> np.ndarray:
    price = np.zeros_like(hour, dtype=float)
    for i, h in enumerate(hour):
        if 0 <= h < 7:
            price[i] = 52.0
        elif 7 <= h < 11:
            price[i] = 88.0
        elif 11 <= h < 16:
            price[i] = 70.0
        elif 16 <= h < 22:
            price[i] = 132.0
        else:
            price[i] = 62.0

    if day_type == "summer_commercial":
        price[(hour >= 13) & (hour < 22)] *= 1.08
    elif day_type == "winter_heating":
        price[((hour >= 7) & (hour < 10)) | ((hour >= 17) & (hour < 22))] *= 1.06
    elif day_type == "spring_irrigation":
        price[(hour >= 8) & (hour < 18)] *= 1.03

    return price


def _renewable_profiles(
    hour: np.ndarray,
    day_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if day_type == "summer_commercial":
        sunrise, sunset, pv_scale, wt_scale = 5.2, 19.4, 1.10, 0.92
    elif day_type == "winter_heating":
        sunrise, sunset, pv_scale, wt_scale = 7.0, 17.2, 0.72, 1.12
    elif day_type == "spring_pv_high":
        sunrise, sunset, pv_scale, wt_scale = 5.8, 18.8, 1.18, 0.95
    elif day_type == "holiday_return":
        sunrise, sunset, pv_scale, wt_scale = 6.7, 17.8, 0.82, 1.06
    else:
        sunrise, sunset, pv_scale, wt_scale = 6.0, 18.5, 1.00, 1.00

    pv_cf = np.zeros_like(hour, dtype=float)
    daylight = (hour >= sunrise) & (hour <= sunset)
    solar_angle = (hour[daylight] - sunrise) / (sunset - sunrise) * np.pi
    pv_cf[daylight] = 0.90 * (np.sin(solar_angle) ** 1.55) * pv_scale
    cloud = (
        1.0
        - 0.08 * _gaussian(hour, 11.5, 0.50)
        - 0.05 * _gaussian(hour, 14.4, 0.65)
        + 0.025 * _gaussian(hour, 12.8, 0.35)
    )
    pv_cf = pv_cf * cloud + _smooth_noise(rng, len(hour), 0.015, 5)
    pv_cf = np.clip(pv_cf, 0.0, 0.96)
    pv_cf[~daylight] = 0.0

    wt_cf = (
        0.40
        + 0.09 * np.sin(2 * np.pi * (hour + 2.0) / 24)
        + 0.055 * np.sin(2 * np.pi * hour / 7.0)
        + _smooth_noise(rng, len(hour), 0.045, 5)
    )
    wt_cf = np.clip(wt_cf * wt_scale, 0.12, 0.82)
    return pv_cf, wt_cf


def generate_one_tongjiang_characteristic_day(
    day_id: int,
    day_type: str,
    day_name: str,
    day_weight: float,
    n_steps_per_hour: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate one Tongjiang-characteristic county representative day.

    The profiles are feature-based synthetic data, not measured Tongjiang operating data.
    They preserve the existing county-scale integrated-energy-system scope while adding
    Tongjiang-relevant agricultural irrigation, grain processing, cold-chain, severe
    winter heating and holiday-return characteristics.
    """

    rng = np.random.default_rng(seed + day_id * 101)
    n = 24 * n_steps_per_hour
    dt = 1.0 / n_steps_per_hour
    hour = np.arange(n, dtype=float) * dt

    # -----------------------------
    # Electric load components / MW
    # -----------------------------
    residential = (
        11.0
        + 3.0 * _gaussian(hour, 7.4, 1.7)
        + 8.5 * _gaussian(hour, 19.4, 2.1)
        + 1.0 * np.sin(2 * np.pi * (hour - 6) / 24)
    )
    commercial = (
        5.5
        + 5.5 * _gaussian(hour, 11.5, 3.2)
        + 5.0 * _gaussian(hour, 18.0, 2.7)
    )
    industrial = 10.5 + 2.2 * _gaussian(hour, 10.0, 3.8) + 2.0 * _gaussian(hour, 16.0, 3.2)
    cold_storage = 4.5 + 1.4 * _gaussian(hour, 14.0, 4.0)
    agro_processing = 1.5 + 1.0 * _gaussian(hour, 13.5, 4.0)
    irrigation = np.zeros_like(hour)

    # -----------------------------
    # Scenario-specific load traits
    # -----------------------------
    heat_scale = 1.0
    cooling_scale = 1.0
    gas_price = 42.0

    if day_type == "spring_irrigation":
        irrigation = 8.5 * ((hour >= 7.0) & (hour < 12.0)) + 6.5 * ((hour >= 13.5) & (hour < 19.0))
        residential *= 0.98
        commercial *= 0.95
        agro_processing *= 1.15
        heat_scale = 0.82
        cooling_scale = 0.62
        gas_price = 41.0

    elif day_type == "spring_pv_high":
        residential *= 0.95
        commercial *= 0.92
        industrial *= 0.92
        cold_storage *= 0.90
        agro_processing *= 0.90
        heat_scale = 0.72
        cooling_scale = 0.55
        gas_price = 40.5

    elif day_type == "summer_commercial":
        residential *= 1.08
        commercial *= 1.30
        cold_storage *= 1.25
        industrial *= 1.03
        heat_scale = 0.42
        cooling_scale = 1.72
        gas_price = 40.0

    elif day_type == "autumn_processing":
        agro_processing = 8.0 + 8.0 * _gaussian(hour, 10.5, 2.8) + 7.0 * _gaussian(hour, 17.0, 3.0)
        industrial *= 1.10
        cold_storage *= 1.15
        residential *= 1.00
        heat_scale = 0.92
        cooling_scale = 0.80
        gas_price = 42.5

    elif day_type == "winter_heating":
        residential *= 1.18
        commercial *= 1.08
        industrial *= 1.04
        cold_storage *= 0.92
        heat_scale = 1.72
        cooling_scale = 0.26
        gas_price = 46.0

    elif day_type == "holiday_return":
        residential *= 1.42
        commercial *= 1.12
        industrial *= 0.72
        agro_processing *= 0.75
        heat_scale = 1.30
        cooling_scale = 0.38
        gas_price = 45.0

    else:
        raise ValueError(f"Unknown Tongjiang-characteristic day type: {day_type}")

    # Critical-load shares are used only for reliability screening.  They do not
    # reduce the normal demand.  Unserved energy, when allowed in stress tests,
    # may only come from the non-critical portion.
    critical_electric_share_map = {
        "spring_irrigation": 0.68,
        "spring_pv_high": 0.58,
        "summer_commercial": 0.62,
        "autumn_processing": 0.70,
        "winter_heating": 0.78,
        "holiday_return": 0.72,
    }
    critical_heat_share_map = {
        "spring_irrigation": 0.75,
        "spring_pv_high": 0.70,
        "summer_commercial": 0.55,
        "autumn_processing": 0.72,
        "winter_heating": 0.92,
        "holiday_return": 0.88,
    }
    critical_cooling_share_map = {
        "spring_irrigation": 0.45,
        "spring_pv_high": 0.40,
        "summer_commercial": 0.68,
        "autumn_processing": 0.62,
        "winter_heating": 0.30,
        "holiday_return": 0.35,
    }

    electric_load = residential + commercial + industrial + cold_storage + agro_processing + irrigation
    electric_load += _smooth_noise(rng, n, 0.85, 7)
    electric_load = np.clip(electric_load, 22.0, None)

    # Heat load / MW
    heat_load = (
        26.0
        + 10.5 * _gaussian(hour, 7.0, 2.2)
        + 7.0 * _gaussian(hour, 19.8, 2.7)
        + 0.20 * agro_processing
        + 1.8 * np.cos(2 * np.pi * (hour - 5.0) / 24)
    )
    heat_load = heat_load * heat_scale + _smooth_noise(rng, n, 0.65, 9)
    heat_load = np.clip(heat_load, 5.0, None)

    # Cooling load / MW
    cooling_load = (
        8.0
        + 13.0 * _gaussian(hour, 14.5, 3.3)
        + 3.5 * _gaussian(hour, 19.0, 2.1)
        + 0.28 * cold_storage
    )
    cooling_load = cooling_load * cooling_scale + _smooth_noise(rng, n, 0.50, 7)
    cooling_load = np.clip(cooling_load, 2.0, None)

    pv_cf, wt_cf = _renewable_profiles(hour, day_type, rng)
    electricity_buy_price = _tou_buy_price(hour, day_type)
    electricity_sell_price = 0.15 * electricity_buy_price

    # Flexible-load availability. This represents timing flexibility, not load shedding.
    shiftable_load_base = (
        0.28 * irrigation
        + 0.18 * agro_processing
        + 0.12 * cold_storage
        + 0.05 * residential
    )
    shiftable_load_base = np.clip(shiftable_load_base, 0.0, 0.18 * electric_load)
    shift_in_capacity = np.clip(0.11 * electric_load, 0.0, None)

    # Interruptible load is disabled in the default scenarios, but the profile is retained
    # for future agricultural/industrial emergency studies.
    interruptible_load_base = 0.08 * irrigation + 0.03 * industrial

    critical_electric_share = np.full(n, critical_electric_share_map[day_type])
    critical_heat_share = np.full(n, critical_heat_share_map[day_type])
    critical_cooling_share = np.full(n, critical_cooling_share_map[day_type])
    critical_electric_load = critical_electric_share * electric_load
    critical_heat_load = critical_heat_share * heat_load
    critical_cooling_load = critical_cooling_share * cooling_load

    # A time-varying grid-availability column makes normal-operation and
    # contingency/stress studies use the same optimization model.
    grid_import_availability = np.ones(n)

    df = pd.DataFrame(
        {
            "day_id": day_id,
            "day_type": day_type,
            "day_name": day_name,
            "day_weight": float(day_weight),
            "case_scope": "同江特征化县域综合能源系统（非实测线路潮流）",
            "data_source_type": "特征化合成数据",
            "tongjiang_feature": {
                "spring_irrigation": "春灌集中用电与农村负荷抬升",
                "spring_pv_high": "春秋低负荷与分布式光伏大发",
                "summer_commercial": "夏季商业制冷与冷链负荷",
                "autumn_processing": "粮食加工、仓储与冷链集中生产",
                "winter_heating": "严寒地区采暖负荷高峰",
                "holiday_return": "春节返乡居民晚峰",
            }[day_type],
            "hour": hour,
            "dt": dt,
            "residential_load": residential,
            "commercial_load": commercial,
            "industrial_load": industrial,
            "cold_storage_load": cold_storage,
            "agro_processing_load": agro_processing,
            "irrigation_load": irrigation,
            "electric_load": electric_load,
            "heat_load": heat_load,
            "cooling_load": cooling_load,
            "critical_electric_share": critical_electric_share,
            "critical_heat_share": critical_heat_share,
            "critical_cooling_share": critical_cooling_share,
            "critical_electric_load": critical_electric_load,
            "critical_heat_load": critical_heat_load,
            "critical_cooling_load": critical_cooling_load,
            "grid_import_availability": grid_import_availability,
            "shiftable_load_base": shiftable_load_base,
            "shift_in_capacity": shift_in_capacity,
            "interruptible_load_base": interruptible_load_base,
            "pv_cf": pv_cf,
            "wt_cf": wt_cf,
            "electricity_buy_price": electricity_buy_price,
            "electricity_sell_price": electricity_sell_price,
            # Compatibility alias for older plotting scripts.
            "electricity_price": electricity_buy_price,
            "gas_price": np.full(n, gas_price),
            "carbon_price": np.full(n, 45.0),
        }
    )
    return df


def generate_tongjiang_characteristic_profiles_v17_3_1(
    n_steps_per_hour: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate six Tongjiang-characteristic county representative days for V17.3.1.

    The profiles are intended for model-method validation and county-level planning
    analysis.  They must not be described as measured Tongjiang feeder data."""

    configs = [
        {"day_id": 0, "day_type": "spring_irrigation", "day_name": "春季灌溉日", "day_weight": 55},
        {"day_id": 1, "day_type": "spring_pv_high", "day_name": "春秋光伏大发日", "day_weight": 65},
        {"day_id": 2, "day_type": "summer_commercial", "day_name": "夏季商业制冷日", "day_weight": 75},
        {"day_id": 3, "day_type": "autumn_processing", "day_name": "秋季农产品加工日", "day_weight": 60},
        {"day_id": 4, "day_type": "winter_heating", "day_name": "冬季采暖日", "day_weight": 80},
        {"day_id": 5, "day_type": "holiday_return", "day_name": "春节返乡日", "day_weight": 30},
    ]

    frames = [
        generate_one_tongjiang_characteristic_day(
            day_id=cfg["day_id"],
            day_type=cfg["day_type"],
            day_name=cfg["day_name"],
            day_weight=cfg["day_weight"],
            n_steps_per_hour=n_steps_per_hour,
            seed=seed,
        )
        for cfg in configs
    ]
    profiles = pd.concat(frames, ignore_index=True)
    profiles["t_global"] = profiles.index * profiles["dt"]
    return profiles


if __name__ == "__main__":
    data = generate_tongjiang_characteristic_profiles_v17_3_1()
    print(data.head())
    print()
    print(data.groupby(["day_id", "day_name"])["day_weight"].first())
    print()
    print(f"Total time steps: {len(data)}")
    print(f"dt: {data['dt'].iloc[0]} h")
    print(f"Annual weight: {data.groupby('day_id')['day_weight'].first().sum()} days")
