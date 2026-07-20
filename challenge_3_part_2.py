"""Challenge 3, Part 2: Orekit propagation with state-selected drag area."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from constants import area_ram_m2, area_tumble_m2, cd, earth_radius_km, mass_kg, mu_earth_km3_s2
from functions import (
    DEG2RAD,
    RAD2DEG,
    SpacecraftState,
    atmospheric_density_kg_m3,
    calibrate_environment_to_decay_rate,
    create_spacecraft,
    propagate_quaternion,
    propagate_schedule,
    update_state,
)


@dataclass
class SimulationResult:
    times_s: np.ndarray
    states: np.ndarray
    operational_states: list[SpacecraftState]
    drag_areas_m2: np.ndarray
    densities_kg_m3: np.ndarray
    masses_kg: np.ndarray


def mission_status(time_s, scenario):
    anomaly_time_s = 6.0 * 3600.0
    recovery_start_s = 30.0 * 3600.0
    recovery_complete_s = 42.0 * 3600.0
    tumble_rate_deg_s = np.array([4.0, -2.0, 6.0])
    if scenario == "nominal" or time_s < anomaly_time_s:
        omega = np.zeros(3); recovering = False; recovered = False
    elif scenario == "temporary_tumble" and time_s < recovery_start_s:
        omega = tumble_rate_deg_s; recovering = False; recovered = False
    elif scenario == "temporary_tumble" and time_s < recovery_complete_s:
        omega = tumble_rate_deg_s * np.exp(-(time_s - recovery_start_s) / 3.0e4)
        recovering = True; recovered = False
    elif scenario == "temporary_tumble":
        omega = np.zeros(3); recovering = False; recovered = True
    elif scenario == "no_recovery":
        omega = tumble_rate_deg_s; recovering = False; recovered = False
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    status = {"omega_mag": float(np.linalg.norm(omega)), "recovering": recovering,
              "recovered": recovered, "mission_lost": False}
    return status, omega * DEG2RAD


def _segments(scenario, duration_s, rho0):
    common = {"cd": cd, "rho0_kg_m3": rho0, "reference_altitude_km": 400.0,
              "scale_height_km": 58.0}
    anomaly = 6.0 * 3600.0
    recovery_start = 30.0 * 3600.0
    recovery_complete = 42.0 * 3600.0
    if scenario == "nominal":
        return [{**common, "duration_s": duration_s, "area_m2": area_ram_m2}]
    if scenario == "temporary_tumble":
        return [
            {**common, "duration_s": anomaly, "area_m2": area_ram_m2},
            {**common, "duration_s": recovery_complete - anomaly, "area_m2": area_tumble_m2},
            {**common, "duration_s": duration_s - recovery_complete, "area_m2": area_ram_m2},
        ]
    return [
        {**common, "duration_s": anomaly, "area_m2": area_ram_m2},
        {**common, "duration_s": duration_s - anomaly, "area_m2": area_tumble_m2},
    ]


def simulate(scenario, duration_days=5.0, step_s=300.0):
    calibration = calibrate_environment_to_decay_rate()
    rho0 = 3.5e-12 * calibration["density_scale"]
    duration_s = duration_days * 86400.0
    radius = earth_radius_km + 400.0
    speed = np.sqrt(mu_earth_km3_s2 / radius)
    initial = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])
    times, translational, masses = propagate_schedule(initial, _segments(scenario, duration_s, rho0), step_s)

    states = np.zeros((len(times), 13))
    states[:, :6] = translational
    states[0, 6] = 1.0
    operational_states = []
    areas = np.zeros(len(times))
    densities = np.zeros(len(times))
    for i, time_s in enumerate(times):
        status, omega = mission_status(time_s, scenario)
        mode = update_state(status)
        operational_states.append(mode)
        areas[i] = area_tumble_m2 if mode in (SpacecraftState.TUMBLING, SpacecraftState.RECOVERY) else area_ram_m2
        states[i, 10:13] = omega
        densities[i] = atmospheric_density_kg_m3(np.linalg.norm(states[i, :3]) - earth_radius_km)
        if i and mode == SpacecraftState.RECOVERED:
            states[i, 6] = 1.0
        elif i:
            dt = times[i] - times[i - 1]
            states[i, 6:10] = propagate_quaternion(states[i - 1, 6:10], omega, dt)
    return SimulationResult(times, states, operational_states, areas, densities, masses)


def semimajor_axis_km(states):
    r = np.linalg.norm(states[:, :3], axis=1)
    v2 = np.sum(states[:, 3:6] ** 2, axis=1)
    return -mu_earth_km3_s2 / (2.0 * (0.5 * v2 - mu_earth_km3_s2 / r))


def main():
    calibration = calibrate_environment_to_decay_rate()
    results = {"Nominal": simulate("nominal"), "Temporary tumble": simulate("temporary_tumble"),
               "No recovery": simulate("no_recovery")}
    nominal_final = results["Nominal"].states[-1, :3]
    spacecraft_cases = {}
    print("Challenge 3, Part 2 — Orekit")
    for label, result in results.items():
        altitude = np.linalg.norm(result.states[:, :3], axis=1) - earth_radius_km
        separation = np.linalg.norm(result.states[-1, :3] - nominal_final)
        print(f"{label:18s}: final altitude={altitude[-1]:.3f} km, minimum={altitude.min():.3f} km, separation={separation:.3f} km")
        spacecraft = create_spacecraft(label)
        spacecraft["orbit"].update({"r": result.states[-1, :3], "v": result.states[-1, 3:6],
            "state_history_km_km_s_quat_rad_s": result.states, "altitude_history_km": altitude,
            "minimum_altitude_km": float(altitude.min()), "final_altitude_km": float(altitude[-1]),
            "final_displacement_from_nominal_km": float(separation),
            "semimajor_axis_history_km": semimajor_axis_km(result.states)})
        spacecraft["vehicle"].update({"mass": float(result.masses_kg[-1]), "cd": cd,
            "drag_area_history_m2": result.drag_areas_m2, "density_history_kg_m3": result.densities_kg_m3})
        spacecraft["operations"].update({"state": result.operational_states[-1],
            "state_history": result.operational_states, "time_history_s": result.times_s,
            "environment_calibration": calibration})
        spacecraft_cases[label] = spacecraft

    plt.figure(figsize=(10, 6))
    for label, result in results.items():
        altitude = np.linalg.norm(result.states[:, :3], axis=1) - earth_radius_km
        plt.plot(result.times_s / 86400.0, altitude, label=label)
    plt.xlabel("Time [days]"); plt.ylabel("Altitude [km]"); plt.title("Orekit State-Dependent Orbital Decay")
    plt.grid(True); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(10, 6))
    for label, result in results.items():
        plt.step(result.times_s / 3600.0, result.drag_areas_m2, where="post", label=label)
    plt.xlabel("Time [h]"); plt.ylabel("Effective drag area [m²]"); plt.title("State-Selected Drag Area")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
    return spacecraft_cases


if __name__ == "__main__":
    main()
