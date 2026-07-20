"""Challenge 3, Part 3: recovery-window and point-of-no-return study.

This script sweeps the time at which recovery begins after an attitude anomaly.
For each case, the spacecraft state machine selects the operational mode and
therefore the effective drag area. The resulting orbit is propagated and tested
against a minimum-altitude requirement.

The objective is to estimate the latest recovery-start time that still satisfies
the mission constraint.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from constants import (
    area_ram_m2,
    area_tumble_m2,
    cd,
    earth_radius_km,
    mass_kg,
    mu_earth_km3_s2,
)
from functions import *


DEG2RAD = np.pi / 180.0


@dataclass
class SweepResult:
    """Summary of one recovery-delay simulation."""

    recovery_start_h: float
    recovery_complete_h: float
    minimum_altitude_km: float
    final_altitude_km: float
    final_separation_km: float
    success: bool



def operational_state(time_s, recovery_start_s, recovery_duration_s):
    """Return the state-machine output for one recovery schedule."""
    anomaly_time_s = 6.0 * 3600.0
    recovery_complete_s = recovery_start_s + recovery_duration_s

    if time_s < anomaly_time_s:
        status = {
            "omega_mag": 0.0,
            "recovering": False,
            "recovered": False,
            "mission_lost": False,
        }
    elif time_s < recovery_start_s:
        status = {
            "omega_mag": np.linalg.norm([4.0, -2.0, 6.0]),
            "recovering": False,
            "recovered": False,
            "mission_lost": False,
        }
    elif time_s < recovery_complete_s:
        status = {
            "omega_mag": 2.0,
            "recovering": True,
            "recovered": False,
            "mission_lost": False,
        }
    else:
        status = {
            "omega_mag": 0.0,
            "recovering": False,
            "recovered": True,
            "mission_lost": False,
        }

    return update_state(status)



def acceleration_km_s2(state, time_s, recovery_start_s, recovery_duration_s):
    """Two-body gravity plus state-dependent atmospheric drag."""
    position_km = state[:3]
    velocity_km_s = state[3:]
    radius_km = np.linalg.norm(position_km)

    gravity_km_s2 = -mu_earth_km3_s2 * position_km / radius_km**3

    mode = operational_state(time_s, recovery_start_s, recovery_duration_s)
    area_m2 = drag_area_from_state(mode)

    altitude_km = radius_km - earth_radius_km
    density_kg_m3 = atmospheric_density_kg_m3(altitude_km)

    position_m = 1000.0 * position_km
    velocity_m_s = 1000.0 * velocity_km_s
    atmosphere_velocity_m_s = np.cross(
        np.array([0.0, 0.0, EARTH_ROTATION_RAD_S]), position_m
    )
    relative_velocity_m_s = velocity_m_s - atmosphere_velocity_m_s
    relative_speed_m_s = np.linalg.norm(relative_velocity_m_s)

    drag_m_s2 = (
        -0.5
        * density_kg_m3
        * cd
        * area_m2
        / mass_kg
        * relative_speed_m_s
        * relative_velocity_m_s
    )

    return gravity_km_s2 + drag_m_s2 / 1000.0


def rk4_step(state, time_s, step_s, recovery_start_s, recovery_duration_s):
    """Advance the Cartesian state by one RK4 step."""

    def derivative(current_state, current_time):
        acceleration = acceleration_km_s2(
            current_state,
            current_time,
            recovery_start_s,
            recovery_duration_s,
        )
        return np.hstack((current_state[3:], acceleration))

    k1 = derivative(state, time_s)
    k2 = derivative(state + 0.5 * step_s * k1, time_s + 0.5 * step_s)
    k3 = derivative(state + 0.5 * step_s * k2, time_s + 0.5 * step_s)
    k4 = derivative(state + step_s * k3, time_s + step_s)

    return state + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def propagate_case(
    recovery_start_h,
    recovery_duration_h=12.0,
    duration_days=10.0,
    step_s=60.0,
):
    """Propagate one recovery schedule and return its Cartesian history."""
    recovery_start_s = recovery_start_h * 3600.0
    recovery_duration_s = recovery_duration_h * 3600.0

    count = int(duration_days * 86400.0 / step_s) + 1
    times_s = np.arange(count) * step_s

    initial_altitude_km = 400.0
    radius_km = earth_radius_km + initial_altitude_km
    speed_km_s = np.sqrt(mu_earth_km3_s2 / radius_km)

    states = np.zeros((count, 6))
    states[0] = np.array([radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0])

    for index in range(count - 1):
        states[index + 1] = rk4_step(
            states[index],
            times_s[index],
            step_s,
            recovery_start_s,
            recovery_duration_s,
        )

    return times_s, states


def evaluate_case(
    recovery_start_h,
    nominal_final_position_km,
    minimum_allowed_altitude_km,
):
    """Propagate and assess one recovery-start time."""
    recovery_duration_h = 12.0
    times_s, states = propagate_case(
        recovery_start_h,
        recovery_duration_h=recovery_duration_h,
    )

    altitude_km = np.linalg.norm(states[:, :3], axis=1) - earth_radius_km
    minimum_altitude_km = float(np.min(altitude_km))
    final_altitude_km = float(altitude_km[-1])
    final_separation_km = float(
        np.linalg.norm(states[-1, :3] - nominal_final_position_km)
    )

    return SweepResult(
        recovery_start_h=float(recovery_start_h),
        recovery_complete_h=float(recovery_start_h + recovery_duration_h),
        minimum_altitude_km=minimum_altitude_km,
        final_altitude_km=final_altitude_km,
        final_separation_km=final_separation_km,
        success=minimum_altitude_km >= minimum_allowed_altitude_km,
    )


def main():
    environment_calibration = calibrate_environment_to_decay_rate()

    minimum_allowed_altitude_km = 390.0

    # A recovery start before the anomaly keeps the spacecraft nominal for this
    # reference case and gives the baseline final position.
    _, nominal_states = propagate_case(recovery_start_h=0.0)
    nominal_final_position_km = nominal_states[-1, :3]

    recovery_start_hours = np.arange(6.0, 192.0 + 6.0, 6.0)
    results = [
        evaluate_case(
            recovery_start_h,
            nominal_final_position_km,
            minimum_allowed_altitude_km,
        )
        for recovery_start_h in recovery_start_hours
    ]

    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]

    spacecraft_cases = []
    for result in results:
        spacecraft = create_spacecraft(
            f"Recovery start {result.recovery_start_h:.1f} h"
        )
        spacecraft["orbit"].update(
            {
                "minimum_allowed_altitude_km": minimum_allowed_altitude_km,
                "minimum_altitude_km": result.minimum_altitude_km,
                "final_altitude_km": result.final_altitude_km,
                "final_separation_from_nominal_km": result.final_separation_km,
            }
        )
        spacecraft["vehicle"].update(
            {
                "mass": mass_kg,
                "cd": cd,
                "nominal_drag_area_m2": area_ram_m2,
                "tumbling_drag_area_m2": area_tumble_m2,
            }
        )
        spacecraft["operations"].update(
            {
                "recovery_start_h": result.recovery_start_h,
                "recovery_complete_h": result.recovery_complete_h,
                "recovered": result.success,
                "mission_lost": not result.success,
                "success": result.success,
                "environment_calibration": environment_calibration,
            }
        )
        spacecraft_cases.append(spacecraft)

    recovery_study = {
        "cases": spacecraft_cases,
        "minimum_allowed_altitude_km": minimum_allowed_altitude_km,
        "recovery_start_hours": recovery_start_hours,
        "environment_calibration": environment_calibration,
    }

    print("Challenge 3, Part 3")
    print("-------------------")
    print(
        "Calibrated density scale: "
        f"{environment_calibration['density_scale']:.6f}"
    )
    print(
        "Reference decay rate:     "
        f"{environment_calibration['calibrated_decay_rate_km_per_day']:.3f} km/day"
    )
    print(f"Minimum permitted altitude: {minimum_allowed_altitude_km:.1f} km")
    print(f"Number of recovery cases:   {len(results)}")

    if successful:
        latest_safe = max(successful, key=lambda result: result.recovery_start_h)
        recovery_study["latest_safe"] = spacecraft_cases[results.index(latest_safe)]
        print(
            "Latest safe recovery start: "
            f"{latest_safe.recovery_start_h:.1f} h after epoch"
        )
        print(
            "Recovery completed at:      "
            f"{latest_safe.recovery_complete_h:.1f} h after epoch"
        )
        print(
            "Minimum altitude in case:   "
            f"{latest_safe.minimum_altitude_km:.3f} km"
        )
    else:
        print("No tested recovery start time satisfied the altitude requirement.")

    if failed:
        first_failed = min(failed, key=lambda result: result.recovery_start_h)
        recovery_study["first_failed"] = spacecraft_cases[results.index(first_failed)]
        print(
            "First failed recovery start: "
            f"{first_failed.recovery_start_h:.1f} h after epoch"
        )
        print(
            "Minimum altitude in case:    "
            f"{first_failed.minimum_altitude_km:.3f} km"
        )
    else:
        print("All tested recovery start times satisfied the altitude requirement.")

    print("\nRecovery sweep:")
    print(" start [h] | complete [h] | min alt [km] | final alt [km] | separation [km] | result")
    for result in results:
        outcome = "PASS" if result.success else "FAIL"
        print(
            f" {result.recovery_start_h:9.1f} |"
            f" {result.recovery_complete_h:12.1f} |"
            f" {result.minimum_altitude_km:12.3f} |"
            f" {result.final_altitude_km:14.3f} |"
            f" {result.final_separation_km:15.3f} | {outcome}"
        )

    minimum_altitudes = np.array(
        [result.minimum_altitude_km for result in results]
    )
    final_altitudes = np.array([result.final_altitude_km for result in results])
    separations = np.array([result.final_separation_km for result in results])
    success_flags = np.array([result.success for result in results], dtype=int)

    plt.figure(figsize=(10, 6))
    plt.plot(recovery_start_hours, minimum_altitudes, marker="o")
    plt.axhline(
        minimum_allowed_altitude_km,
        linestyle="--",
        label="Minimum allowed altitude",
    )
    plt.xlabel("Recovery-start time after epoch [h]")
    plt.ylabel("Minimum altitude over propagation [km]")
    plt.title("Recovery Window and Minimum Altitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(recovery_start_hours, final_altitudes, marker="o")
    plt.xlabel("Recovery-start time after epoch [h]")
    plt.ylabel("Final altitude [km]")
    plt.title("Final Altitude Versus Recovery Delay")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(recovery_start_hours, separations, marker="o")
    plt.xlabel("Recovery-start time after epoch [h]")
    plt.ylabel("Final position separation from nominal [km]")
    plt.title("Orbital Consequence of Delayed Recovery")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.step(recovery_start_hours, success_flags, where="mid")
    plt.yticks([0, 1], ["Fail", "Pass"])
    plt.xlabel("Recovery-start time after epoch [h]")
    plt.ylabel("Mission outcome")
    plt.title("Recovery Success/Failure Boundary")
    plt.grid(True)
    plt.tight_layout()

    if successful and failed:
        latest_safe_time = max(result.recovery_start_h for result in successful)
        first_failed_time = min(result.recovery_start_h for result in failed)

        representative_cases = {
            "Latest safe": latest_safe_time,
            "First failed": first_failed_time,
        }

        plt.figure(figsize=(10, 6))
        for label, recovery_start_h in representative_cases.items():
            times_s, states = propagate_case(recovery_start_h)
            altitude_km = np.linalg.norm(states[:, :3], axis=1) - earth_radius_km
            plt.plot(times_s / 86400.0, altitude_km, label=label)

        plt.axhline(
            minimum_allowed_altitude_km,
            linestyle="--",
            label="Minimum allowed altitude",
        )
        plt.xlabel("Time [days]")
        plt.ylabel("Altitude [km]")
        plt.title("Representative Safe and Failed Recovery Cases")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    plt.show()

    return recovery_study


if __name__ == "__main__":
    main()
