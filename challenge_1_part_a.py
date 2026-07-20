"""Challenge 1, Part A: Orekit RAAN-drift and finite-HET deployment trade."""

import matplotlib.pyplot as plt
import numpy as np

from constants import (
    area_ram_m2, cd, drift_duration_days, earth_radius_km, inclination_deg,
    initial_altitude_km, isp_s, mass_kg, mu_earth_km3_s2,
    strategy_a_raise_delta_v_mps, strategy_b_drift_delta_v_mps,
    strategy_b_raise_delta_v_mps, target_altitude_km, thrust_n,
)
from functions import calibrate_environment_to_decay_rate, create_spacecraft, propagate_schedule


def circular_state(altitude_km, inclination_degrees=inclination_deg):
    radius = earth_radius_km + altitude_km
    speed = np.sqrt(mu_earth_km3_s2 / radius)
    inclination = np.radians(inclination_degrees)
    return np.array([radius, 0.0, 0.0, 0.0, speed * np.cos(inclination), speed * np.sin(inclination)])


def orbital_history(states):
    r = states[:, :3]
    v = states[:, 3:]
    radius = np.linalg.norm(r, axis=1)
    h = np.cross(r, v)
    node = np.cross(np.tile([0.0, 0.0, 1.0], (len(states), 1)), h)
    raan = np.unwrap(np.arctan2(node[:, 1], node[:, 0]))
    energy = 0.5 * np.sum(v * v, axis=1) - mu_earth_km3_s2 / radius
    semi_major_axis = -mu_earth_km3_s2 / (2.0 * energy)
    return semi_major_axis - earth_radius_km, np.degrees(raan - raan[0])


def _environment():
    calibration = calibrate_environment_to_decay_rate()
    return calibration, {
        "cd": cd,
        "rho0_kg_m3": 3.5e-12 * calibration["density_scale"],
        "reference_altitude_km": 400.0,
        # The challenge supplies an approximately constant 330 km decay rate
        # and a 273.7 km endpoint after 110 days.  Use a broad calibrated
        # density scale here so Orekit reproduces that campaign assumption
        # instead of imposing an unrelated epoch-specific thermosphere.
        "scale_height_km": 1000.0,
    }


def _maintenance_segments(common):
    """Distribute the specified station-keeping delta-v over the 110-day drift."""
    total_burn_s = strategy_b_drift_delta_v_mps * mass_kg / thrust_n
    control_interval_days = 10
    intervals = int(np.ceil(drift_duration_days / control_interval_days))
    burn_per_interval = total_burn_s / intervals
    segments = []
    for interval in range(intervals):
        interval_days = min(control_interval_days, drift_duration_days - interval * control_interval_days)
        coast = interval_days * 86400.0 - burn_per_interval
        if coast > 0.0:
            segments.append({**common, "duration_s": coast, "area_m2": area_ram_m2})
        segments.append({**common, "duration_s": burn_per_interval, "area_m2": area_ram_m2,
                         "thrust_n": thrust_n, "isp_s": isp_s, "thrust_direction": "PROGRADE"})
    return segments


def _raise_segment(common, delta_v_mps):
    # The challenge specifies campaign delta-v; Orekit applies it as a finite HET arc.
    duration = delta_v_mps * mass_kg / thrust_n
    return {**common, "duration_s": duration, "area_m2": area_ram_m2,
            "thrust_n": thrust_n, "isp_s": isp_s, "thrust_direction": "PROGRADE"}


def main():
    calibration, common = _environment()
    initial = circular_state(initial_altitude_km)
    drift_s = drift_duration_days * 86400.0

    a_times, a_drift_states, a_drift_mass = propagate_schedule(
        initial, [{**common, "duration_s": drift_s, "area_m2": area_ram_m2}], 21600.0)
    a_raise_times, a_raise_states, a_raise_mass = propagate_schedule(
        a_drift_states[-1], [_raise_segment(common, strategy_a_raise_delta_v_mps)], 21600.0,
        initial_mass_kg=float(a_drift_mass[-1]))
    a_all_times = np.concatenate((a_times, a_times[-1] + a_raise_times[1:]))
    a_all_states = np.vstack((a_drift_states, a_raise_states[1:]))
    a_all_mass = np.concatenate((a_drift_mass, a_raise_mass[1:]))

    b_segments = _maintenance_segments(common)
    b_times, b_drift_states, b_drift_mass = propagate_schedule(initial, b_segments, 21600.0)
    b_raise_times, b_raise_states, b_raise_mass = propagate_schedule(
        b_drift_states[-1], [_raise_segment(common, strategy_b_raise_delta_v_mps)], 21600.0,
        initial_mass_kg=float(b_drift_mass[-1]))
    b_all_times = np.concatenate((b_times, b_times[-1] + b_raise_times[1:]))
    b_all_states = np.vstack((b_drift_states, b_raise_states[1:]))
    b_all_mass = np.concatenate((b_drift_mass, b_raise_mass[1:]))

    a_alt, a_raan = orbital_history(a_drift_states)
    b_alt, b_raan = orbital_history(b_drift_states)
    b_raan_on_a_times = np.interp(a_times, b_times, b_raan)
    relative_raan = a_raan - b_raan_on_a_times

    print("Challenge 1, Part A — Orekit")
    print(f"Strategy A altitude after drift: {a_alt[-1]:.3f} km")
    print(f"Strategy B altitude after drift: {b_alt[-1]:.3f} km")
    print(f"Relative RAAN accumulated:       {relative_raan[-1]:.3f} deg")
    print(f"Strategy A final mass:           {a_all_mass[-1]:.3f} kg")
    print(f"Strategy B final mass:           {b_all_mass[-1]:.3f} kg")
    print(f"Strategy A post-raise altitude:  {orbital_history(a_all_states)[0][-1]:.3f} km")
    print(f"Strategy B post-raise altitude:  {orbital_history(b_all_states)[0][-1]:.3f} km")

    plt.figure(figsize=(10, 6))
    plt.plot(a_times / 86400.0, a_alt, label="Strategy A: passive decay")
    plt.plot(b_times / 86400.0, b_alt, label="Strategy B: maintained altitude")
    plt.axhline(target_altitude_km, linestyle="--", label="Operational altitude")
    plt.xlabel("Time [days]"); plt.ylabel("Osculating semimajor-axis altitude [km]")
    plt.title("Orekit Drift-Orbit Evolution"); plt.grid(True); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(a_times / 86400.0, a_raan, label="Strategy A")
    plt.plot(b_times / 86400.0, b_raan, label="Strategy B")
    plt.plot(a_times / 86400.0, relative_raan, label="A minus B")
    plt.xlabel("Time [days]"); plt.ylabel("RAAN change [deg]")
    plt.title("Orekit RAAN Evolution"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    result = {"Strategy A": create_spacecraft("Strategy A"), "Strategy B": create_spacecraft("Strategy B")}
    result["Strategy A"]["orbit"].update({"state_history": a_all_states, "time_history_s": a_all_times,
        "drift_altitude_history_km": a_alt, "drift_raan_history_deg": a_raan})
    result["Strategy B"]["orbit"].update({"state_history": b_all_states, "time_history_s": b_all_times,
        "drift_altitude_history_km": b_alt, "drift_raan_history_deg": b_raan})
    result["Strategy A"]["vehicle"].update({"mass": float(a_all_mass[-1]), "cd": cd, "drag_area": area_ram_m2})
    result["Strategy B"]["vehicle"].update({"mass": float(b_all_mass[-1]), "cd": cd, "drag_area": area_ram_m2})
    result["environment_calibration"] = calibration
    return result


if __name__ == "__main__":
    main()
