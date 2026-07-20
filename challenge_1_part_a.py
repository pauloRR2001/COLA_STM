"""Challenge 1, Part A: Orekit RAAN-drift and finite-HET deployment trade."""

import matplotlib.pyplot as plt
import numpy as np

from constants import (
    area_ram_m2,
    cd,
    drift_duration_days,
    earth_radius_km,
    inclination_deg,
    initial_altitude_km,
    isp_s,
    mass_kg,
    mu_earth_km3_s2,
    passive_final_altitude_km,
    target_altitude_km,
    thrust_n,
)
from functions import calibrate_environment_to_decay_rate, create_spacecraft, propagate_schedule


def circular_state(altitude_km, inclination_degrees=inclination_deg):
    radius = earth_radius_km + altitude_km
    speed = np.sqrt(mu_earth_km3_s2 / radius)
    inclination = np.radians(inclination_degrees)
    return np.array(
        [radius, 0.0, 0.0, 0.0, speed * np.cos(inclination), speed * np.sin(inclination)]
    )


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


def mean_sma_altitude(states, tail_points=24):
    altitude, _ = orbital_history(states)
    return float(np.mean(altitude[-min(tail_points, len(altitude)) :]))


def _environment():
    calibration = calibrate_environment_to_decay_rate(
        reference_altitude_km=initial_altitude_km,
        calibration_duration_days=drift_duration_days,
        target_final_altitude_km=passive_final_altitude_km,
        reference_altitude_model_km=400.0,
        scale_height_km=1000.0,
    )
    return calibration, {
        "cd": cd,
        "rho0_kg_m3": 3.5e-12 * calibration["density_scale"],
        "reference_altitude_km": 400.0,
        "scale_height_km": 1000.0,
    }


def solve_continuous_thrust(
    common,
    initial_state,
    duration_s,
    target_altitude,
    initial_mass_kg=mass_kg,
    max_thrust_n=thrust_n,
):
    """Solve an equivalent continuous HET thrust level that holds the target orbit."""

    def propagate(thrust_level):
        return propagate_schedule(
            initial_state,
            [
                {
                    **common,
                    "duration_s": duration_s,
                    "area_m2": area_ram_m2,
                    "thrust_n": thrust_level,
                    "isp_s": isp_s,
                    "thrust_direction": "PROGRADE",
                }
            ],
            21600.0,
            initial_mass_kg=initial_mass_kg,
        )

    low = 0.0
    high = max_thrust_n
    if mean_sma_altitude(propagate(high)[1]) < target_altitude:
        raise RuntimeError("The available HET thrust cannot maintain the requested orbit")

    for _ in range(28):
        trial = 0.5 * (low + high)
        result = propagate(trial)
        if mean_sma_altitude(result[1]) < target_altitude:
            low = trial
        else:
            high = trial

    thrust_level = 0.5 * (low + high)
    return thrust_level, propagate(thrust_level)


def solve_raise_duration(common, initial_state, initial_mass_kg):
    """Solve full-thrust burn duration to terminate at the 520 km target orbit."""

    def propagate(duration_s):
        return propagate_schedule(
            initial_state,
            [
                {
                    **common,
                    "duration_s": duration_s,
                    "area_m2": area_ram_m2,
                    "thrust_n": thrust_n,
                    "isp_s": isp_s,
                    "thrust_direction": "PROGRADE",
                }
            ],
            3600.0,
            initial_mass_kg=initial_mass_kg,
        )

    low = 1.0
    high = 24.0 * 86400.0
    while mean_sma_altitude(propagate(high)[1]) < target_altitude_km:
        high *= 1.5
        if high > 180.0 * 86400.0:
            raise RuntimeError("Unable to bracket the 520 km orbit-raise duration")

    for _ in range(28):
        trial = 0.5 * (low + high)
        result = propagate(trial)
        if mean_sma_altitude(result[1]) < target_altitude_km:
            low = trial
        else:
            high = trial

    duration_s = 0.5 * (low + high)
    return duration_s, propagate(duration_s)


def main():
    calibration, common = _environment()
    initial = circular_state(initial_altitude_km)
    drift_s = drift_duration_days * 86400.0

    a_times, a_drift_states, a_drift_mass = propagate_schedule(
        initial,
        [{**common, "duration_s": drift_s, "area_m2": area_ram_m2}],
        21600.0,
    )
    a_raise_duration, (a_raise_times, a_raise_states, a_raise_mass) = solve_raise_duration(
        common, a_drift_states[-1], float(a_drift_mass[-1])
    )
    a_all_times = np.concatenate((a_times, a_times[-1] + a_raise_times[1:]))
    a_all_states = np.vstack((a_drift_states, a_raise_states[1:]))
    a_all_mass = np.concatenate((a_drift_mass, a_raise_mass[1:]))

    maintenance_thrust, (b_times, b_drift_states, b_drift_mass) = solve_continuous_thrust(
        common, initial, drift_s, initial_altitude_km
    )
    b_raise_duration, (b_raise_times, b_raise_states, b_raise_mass) = solve_raise_duration(
        common, b_drift_states[-1], float(b_drift_mass[-1])
    )
    b_all_times = np.concatenate((b_times, b_times[-1] + b_raise_times[1:]))
    b_all_states = np.vstack((b_drift_states, b_raise_states[1:]))
    b_all_mass = np.concatenate((b_drift_mass, b_raise_mass[1:]))

    a_alt, a_raan = orbital_history(a_drift_states)
    b_alt, b_raan = orbital_history(b_drift_states)
    relative_raan = a_raan - np.interp(a_times, b_times, b_raan)

    maintenance_delta_v = maintenance_thrust / mass_kg * drift_s
    a_raise_delta_v = thrust_n / mass_kg * a_raise_duration
    b_raise_delta_v = thrust_n / mass_kg * b_raise_duration

    print("Challenge 1, Part A — Orekit")
    print(f"Strategy A altitude after drift: {mean_sma_altitude(a_drift_states):.3f} km")
    print(f"Strategy B altitude after drift: {mean_sma_altitude(b_drift_states):.3f} km")
    print(f"Relative RAAN accumulated:       {relative_raan[-1]:.3f} deg")
    print(f"Strategy A post-raise altitude:  {mean_sma_altitude(a_raise_states):.3f} km")
    print(f"Strategy B post-raise altitude:  {mean_sma_altitude(b_raise_states):.3f} km")
    print(f"Strategy B continuous thrust:    {maintenance_thrust * 1000.0:.3f} mN")
    print(f"Strategy B equivalent duty cycle:{maintenance_thrust / thrust_n:.4f}")
    print(f"Strategy B drift delta-v:        {maintenance_delta_v:.3f} m/s")
    print(f"Strategy A raise delta-v:        {a_raise_delta_v:.3f} m/s")
    print(f"Strategy B raise delta-v:        {b_raise_delta_v:.3f} m/s")
    print(f"Strategy A total delta-v:        {a_raise_delta_v:.3f} m/s")
    print(f"Strategy B total delta-v:        {maintenance_delta_v + b_raise_delta_v:.3f} m/s")

    plt.figure(figsize=(10, 6))
    plt.plot(a_times / 86400.0, a_alt, label="Strategy A: passive decay")
    plt.plot(b_times / 86400.0, b_alt, label="Strategy B: maintained altitude")
    plt.axhline(target_altitude_km, linestyle="--", label="Operational altitude")
    plt.xlabel("Time [days]")
    plt.ylabel("Osculating semimajor-axis altitude [km]")
    plt.title("Orekit Drift-Orbit Evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(a_times / 86400.0, a_raan, label="Strategy A")
    plt.plot(b_times / 86400.0, b_raan, label="Strategy B")
    plt.plot(a_times / 86400.0, relative_raan, label="A minus B")
    plt.xlabel("Time [days]")
    plt.ylabel("RAAN change [deg]")
    plt.title("Orekit RAAN Evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    result = {
        "Strategy A": create_spacecraft("Strategy A"),
        "Strategy B": create_spacecraft("Strategy B"),
    }
    result["Strategy A"]["orbit"].update(
        {
            "state_history": a_all_states,
            "time_history_s": a_all_times,
            "drift_altitude_history_km": a_alt,
            "drift_raan_history_deg": a_raan,
        }
    )
    result["Strategy B"]["orbit"].update(
        {
            "state_history": b_all_states,
            "time_history_s": b_all_times,
            "drift_altitude_history_km": b_alt,
            "drift_raan_history_deg": b_raan,
        }
    )
    result["Strategy A"]["vehicle"].update(
        {"mass": float(a_all_mass[-1]), "cd": cd, "drag_area": area_ram_m2}
    )
    result["Strategy B"]["vehicle"].update(
        {"mass": float(b_all_mass[-1]), "cd": cd, "drag_area": area_ram_m2}
    )
    result["environment_calibration"] = calibration
    return result


if __name__ == "__main__":
    main()
