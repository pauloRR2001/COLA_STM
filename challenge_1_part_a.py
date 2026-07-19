"""Challenge 1, Part A: RAAN drift and campaign delta-v trade.

Compares a naturally decaying drift orbit (Strategy A) with altitude
maintenance at the deployment altitude (Strategy B), then reports the
campaign delta-v accounting supplied with the challenge.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from constants import (
    drift_duration_days,
    earth_radius_km,
    inclination_deg,
    initial_altitude_km,
    j2,
    mu_earth_km3_s2,
    nominal_decay_rate_km_per_day,
    passive_final_altitude_km,
    seconds_per_day,
    strategy_a_delta_v_savings_mps,
    strategy_a_drift_delta_v_mps,
    strategy_a_raise_delta_v_mps,
    strategy_a_total_delta_v_mps,
    strategy_b_drift_delta_v_mps,
    strategy_b_raise_delta_v_mps,
    strategy_b_total_delta_v_mps,
    target_altitude_km,
)


def raan_rate_rad_s(altitude_km: np.ndarray | float) -> np.ndarray | float:
    """First-order secular J2 RAAN rate for a circular orbit."""
    semi_major_axis_km = earth_radius_km + np.asarray(altitude_km)
    mean_motion_rad_s = np.sqrt(mu_earth_km3_s2 / semi_major_axis_km**3)
    return (
        -1.5
        * j2
        * mean_motion_rad_s
        * (earth_radius_km / semi_major_axis_km) ** 2
        * np.cos(np.radians(inclination_deg))
    )


def hohmann_delta_v_mps(initial_altitude_km_value: float, final_altitude_km_value: float) -> float:
    """Two-impulse circular-to-circular Hohmann transfer delta-v."""
    r1 = earth_radius_km + initial_altitude_km_value
    r2 = earth_radius_km + final_altitude_km_value
    transfer_axis = 0.5 * (r1 + r2)
    v1 = math.sqrt(mu_earth_km3_s2 / r1)
    v2 = math.sqrt(mu_earth_km3_s2 / r2)
    vt1 = math.sqrt(mu_earth_km3_s2 * (2.0 / r1 - 1.0 / transfer_axis))
    vt2 = math.sqrt(mu_earth_km3_s2 * (2.0 / r2 - 1.0 / transfer_axis))
    return 1000.0 * (abs(vt1 - v1) + abs(v2 - vt2))


def main() -> None:
    time_days = np.arange(drift_duration_days + 1, dtype=float)

    # Strategy A follows the specified decay law, clipped at the stated final altitude.
    altitude_a_km = initial_altitude_km + nominal_decay_rate_km_per_day * time_days
    altitude_a_km = np.maximum(altitude_a_km, passive_final_altitude_km)

    # Strategy B maintains the deployment altitude throughout the drift campaign.
    altitude_b_km = np.full_like(time_days, initial_altitude_km)

    rate_a_deg_day = np.degrees(raan_rate_rad_s(altitude_a_km)) * seconds_per_day
    rate_b_deg_day = np.degrees(raan_rate_rad_s(altitude_b_km)) * seconds_per_day

    # Trapezoidal integration avoids the one-day offset in the original script.
    raan_a_deg = np.zeros_like(time_days)
    raan_b_deg = np.zeros_like(time_days)
    raan_a_deg[1:] = np.cumsum(0.5 * (rate_a_deg_day[:-1] + rate_a_deg_day[1:]))
    raan_b_deg[1:] = np.cumsum(0.5 * (rate_b_deg_day[:-1] + rate_b_deg_day[1:]))

    raan_separation_deg = raan_a_deg - raan_b_deg

    impulsive_raise_a_mps = hohmann_delta_v_mps(
        passive_final_altitude_km, target_altitude_km
    )
    impulsive_raise_b_mps = hohmann_delta_v_mps(
        initial_altitude_km, target_altitude_km
    )

    print("Challenge 1 Part A — RAAN drift and campaign trade")
    print(f"Final Strategy A altitude:       {altitude_a_km[-1]:.3f} km")
    print(f"Final Strategy B altitude:       {altitude_b_km[-1]:.3f} km")
    print(f"Final Strategy A RAAN change:    {raan_a_deg[-1]:.6f} deg")
    print(f"Final Strategy B RAAN change:    {raan_b_deg[-1]:.6f} deg")
    print(f"Final RAAN separation (A-B):     {raan_separation_deg[-1]:.6f} deg")
    print()
    print("Reference impulsive transfer values (Hohmann):")
    print(f"  {passive_final_altitude_km:.1f} -> {target_altitude_km:.1f} km: {impulsive_raise_a_mps:.3f} m/s")
    print(f"  {initial_altitude_km:.1f} -> {target_altitude_km:.1f} km: {impulsive_raise_b_mps:.3f} m/s")
    print()
    print("Challenge campaign delta-v accounting:")
    print(f"  Strategy A drift:              {strategy_a_drift_delta_v_mps:.1f} m/s")
    print(f"  Strategy A raise:              {strategy_a_raise_delta_v_mps:.1f} m/s")
    print(f"  Strategy A total:              {strategy_a_total_delta_v_mps:.1f} m/s")
    print(f"  Strategy B drift:              {strategy_b_drift_delta_v_mps:.1f} m/s")
    print(f"  Strategy B raise:              {strategy_b_raise_delta_v_mps:.1f} m/s")
    print(f"  Strategy B total:              {strategy_b_total_delta_v_mps:.1f} m/s")
    print(f"  Strategy A savings:            {strategy_a_delta_v_savings_mps:.1f} m/s")

    plt.figure()
    plt.plot(time_days, altitude_a_km, label="Strategy A: passive decay")
    plt.plot(time_days, altitude_b_km, label="Strategy B: maintained altitude")
    plt.xlabel("Time [days]")
    plt.ylabel("Altitude [km]")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(time_days, rate_a_deg_day, label="Strategy A")
    plt.plot(time_days, rate_b_deg_day, label="Strategy B")
    plt.xlabel("Time [days]")
    plt.ylabel("RAAN rate [deg/day]")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(time_days, raan_a_deg, label="Strategy A")
    plt.plot(time_days, raan_b_deg, label="Strategy B")
    plt.xlabel("Time [days]")
    plt.ylabel("Accumulated RAAN change [deg]")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(time_days, raan_separation_deg)
    plt.xlabel("Time [days]")
    plt.ylabel("RAAN separation, A - B [deg]")
    plt.grid(True)

    plt.figure()
    labels = ["A drift", "A raise", "B drift", "B raise"]
    values = [
        strategy_a_drift_delta_v_mps,
        strategy_a_raise_delta_v_mps,
        strategy_b_drift_delta_v_mps,
        strategy_b_raise_delta_v_mps,
    ]
    plt.bar(labels, values)
    plt.ylabel("Delta-v [m/s]")
    plt.title("Campaign delta-v components")
    plt.grid(True, axis="y")

    plt.show()


if __name__ == "__main__":
    main()
