"""Challenge 3, Part 3: Orekit recovery-window and point-of-no-return sweep."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from constants import area_ram_m2, area_tumble_m2, cd, earth_radius_km, mass_kg, mu_earth_km3_s2
from functions import calibrate_environment_to_decay_rate, create_spacecraft, propagate_schedule


@dataclass
class SweepResult:
    recovery_start_h: float
    recovery_complete_h: float
    minimum_altitude_km: float
    final_altitude_km: float
    final_separation_km: float
    success: bool


def propagate_case(recovery_start_h, recovery_duration_h=12.0, duration_days=10.0, step_s=600.0):
    """Propagate one recovery schedule entirely with Orekit force models."""
    calibration = calibrate_environment_to_decay_rate()
    rho0 = 3.5e-12 * calibration["density_scale"]
    anomaly_s = 6.0 * 3600.0
    recovery_start_s = max(anomaly_s, recovery_start_h * 3600.0)
    recovery_complete_s = min(duration_days * 86400.0, recovery_start_s + recovery_duration_h * 3600.0)
    duration_s = duration_days * 86400.0
    common = {"cd": cd, "rho0_kg_m3": rho0, "reference_altitude_km": 400.0,
              "scale_height_km": 58.0}
    segments = [{**common, "duration_s": anomaly_s, "area_m2": area_ram_m2}]
    if recovery_start_s > anomaly_s:
        segments.append({**common, "duration_s": recovery_start_s - anomaly_s, "area_m2": area_tumble_m2})
    if recovery_complete_s > recovery_start_s:
        segments.append({**common, "duration_s": recovery_complete_s - recovery_start_s, "area_m2": area_tumble_m2})
    if duration_s > recovery_complete_s:
        segments.append({**common, "duration_s": duration_s - recovery_complete_s, "area_m2": area_ram_m2})

    radius = earth_radius_km + 400.0
    speed = np.sqrt(mu_earth_km3_s2 / radius)
    initial = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])
    times, states, _ = propagate_schedule(initial, segments, step_s)
    return times, states


def evaluate_case(recovery_start_h, nominal_final_position_km, minimum_allowed_altitude_km):
    recovery_duration_h = 12.0
    _, states = propagate_case(recovery_start_h, recovery_duration_h)
    altitude = np.linalg.norm(states[:, :3], axis=1) - earth_radius_km
    return SweepResult(float(recovery_start_h), float(recovery_start_h + recovery_duration_h),
        float(altitude.min()), float(altitude[-1]),
        float(np.linalg.norm(states[-1, :3] - nominal_final_position_km)),
        bool(altitude.min() >= minimum_allowed_altitude_km))


def main():
    calibration = calibrate_environment_to_decay_rate()
    minimum_allowed_altitude_km = 390.0
    _, nominal_states = propagate_case(0.0)
    nominal_final = nominal_states[-1, :3]
    recovery_start_hours = np.arange(6.0, 192.0 + 6.0, 6.0)
    results = [evaluate_case(t, nominal_final, minimum_allowed_altitude_km) for t in recovery_start_hours]
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    spacecraft_cases = []
    for result in results:
        spacecraft = create_spacecraft(f"Recovery start {result.recovery_start_h:.1f} h")
        spacecraft["orbit"].update({"minimum_allowed_altitude_km": minimum_allowed_altitude_km,
            "minimum_altitude_km": result.minimum_altitude_km, "final_altitude_km": result.final_altitude_km,
            "final_separation_from_nominal_km": result.final_separation_km})
        spacecraft["vehicle"].update({"mass": mass_kg, "cd": cd,
            "nominal_drag_area_m2": area_ram_m2, "tumbling_drag_area_m2": area_tumble_m2})
        spacecraft["operations"].update({"recovery_start_h": result.recovery_start_h,
            "recovery_complete_h": result.recovery_complete_h, "recovered": result.success,
            "mission_lost": not result.success, "success": result.success,
            "environment_calibration": calibration, "propagator": "Orekit NumericalPropagator"})
        spacecraft_cases.append(spacecraft)

    study = {"cases": spacecraft_cases, "minimum_allowed_altitude_km": minimum_allowed_altitude_km,
             "recovery_start_hours": recovery_start_hours, "environment_calibration": calibration}
    print("Challenge 3, Part 3 — Orekit")
    if successful:
        latest = max(successful, key=lambda r: r.recovery_start_h)
        study["latest_safe"] = spacecraft_cases[results.index(latest)]
        print(f"Latest safe recovery start: {latest.recovery_start_h:.1f} h")
    if failed:
        first = min(failed, key=lambda r: r.recovery_start_h)
        study["first_failed"] = spacecraft_cases[results.index(first)]
        print(f"First failed recovery start: {first.recovery_start_h:.1f} h")

    minimum_altitudes = np.array([r.minimum_altitude_km for r in results])
    final_altitudes = np.array([r.final_altitude_km for r in results])
    separations = np.array([r.final_separation_km for r in results])
    flags = np.array([r.success for r in results], dtype=int)

    plt.figure(figsize=(10, 6)); plt.plot(recovery_start_hours, minimum_altitudes, marker="o")
    plt.axhline(minimum_allowed_altitude_km, linestyle="--", label="Minimum allowed altitude")
    plt.xlabel("Recovery-start time [h]"); plt.ylabel("Minimum altitude [km]")
    plt.title("Orekit Recovery Window"); plt.grid(True); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(10, 6)); plt.plot(recovery_start_hours, final_altitudes, marker="o")
    plt.xlabel("Recovery-start time [h]"); plt.ylabel("Final altitude [km]")
    plt.title("Final Altitude Versus Recovery Delay"); plt.grid(True); plt.tight_layout()

    plt.figure(figsize=(10, 6)); plt.plot(recovery_start_hours, separations, marker="o")
    plt.xlabel("Recovery-start time [h]"); plt.ylabel("Final separation from nominal [km]")
    plt.title("Orbital Consequence of Delayed Recovery"); plt.grid(True); plt.tight_layout()

    plt.figure(figsize=(10, 4)); plt.step(recovery_start_hours, flags, where="mid")
    plt.yticks([0, 1], ["Fail", "Pass"]); plt.xlabel("Recovery-start time [h]")
    plt.ylabel("Mission outcome"); plt.title("Recovery Success Boundary"); plt.grid(True); plt.tight_layout()
    plt.show()
    return study


if __name__ == "__main__":
    main()
