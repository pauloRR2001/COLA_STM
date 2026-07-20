"""Challenge 3, Part 2: coupled orbit, attitude, and operational states.

Three cases are compared:

1. Nominal attitude for the complete propagation.
2. A temporary tumble followed by recovery.
3. A tumble with no recovery.

The simulation propagates position, velocity, quaternion, and angular velocity.
The state machine selects the operational mode, and that mode selects the
spacecraft drag area. ADCS hardware is intentionally not modeled; angular-rate
profiles are prescribed at the mission-operations level.
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




@dataclass
class SimulationResult:
    """History produced by one mission scenario."""

    times_s: np.ndarray
    states: np.ndarray
    operational_states: list[SpacecraftState]
    drag_areas_m2: np.ndarray
    densities_kg_m3: np.ndarray





def mission_status(time_s, scenario):
    """Build the status dictionary consumed by the shared state machine."""
    anomaly_time_s = 6.0 * 3600.0
    recovery_start_s = 30.0 * 3600.0
    recovery_complete_s = 42.0 * 3600.0
    tumble_rate_deg_s = np.array([4.0, -2.0, 6.0])

    if scenario == "nominal" or time_s < anomaly_time_s:
        omega_deg_s = np.zeros(3)
        recovering = False
        recovered = False
    elif scenario == "temporary_tumble":
        if time_s < recovery_start_s:
            omega_deg_s = tumble_rate_deg_s
            recovering = False
            recovered = False
        elif time_s < recovery_complete_s:
            elapsed_s = time_s - recovery_start_s
            omega_deg_s = tumble_rate_deg_s * np.exp(-elapsed_s / 3.0e4)
            recovering = True
            recovered = False
        else:
            omega_deg_s = np.zeros(3)
            recovering = False
            recovered = True
    elif scenario == "no_recovery":
        omega_deg_s = tumble_rate_deg_s
        recovering = False
        recovered = False
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    status = {
        "omega_mag": float(np.linalg.norm(omega_deg_s)),
        "recovering": recovering,
        "recovered": recovered,
        "mission_lost": False,
    }
    return status, omega_deg_s * DEG2RAD



def acceleration_km_s2(translational_state, time_s, scenario):
    """Two-body gravity plus state-dependent atmospheric drag."""
    position_km = translational_state[:3]
    velocity_km_s = translational_state[3:]
    radius_km = np.linalg.norm(position_km)

    gravity_km_s2 = -mu_earth_km3_s2 * position_km / radius_km**3

    status, _ = mission_status(time_s, scenario)
    operational_state = update_state(status)
    area_m2 = drag_area_from_state(operational_state)

    altitude_km = radius_km - earth_radius_km
    density_kg_m3 = atmospheric_density_kg_m3(altitude_km)

    position_m = position_km * 1000.0
    velocity_m_s = velocity_km_s * 1000.0
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


def rk4_translation_step(state, time_s, step_s, scenario):
    """Advance the six-element translational state with RK4."""

    def derivative(current_state, current_time):
        return np.hstack(
            (
                current_state[3:],
                acceleration_km_s2(current_state, current_time, scenario),
            )
        )

    k1 = derivative(state, time_s)
    k2 = derivative(state + 0.5 * step_s * k1, time_s + 0.5 * step_s)
    k3 = derivative(state + 0.5 * step_s * k2, time_s + 0.5 * step_s)
    k4 = derivative(state + step_s * k3, time_s + step_s)
    return state + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate(scenario, duration_days=5.0, step_s=30.0):
    """Propagate one coupled orbit-attitude mission scenario."""
    count = int(duration_days * 86400.0 / step_s) + 1
    times_s = np.arange(count) * step_s

    altitude_km = 400.0
    radius_km = earth_radius_km + altitude_km
    speed_km_s = np.sqrt(mu_earth_km3_s2 / radius_km)

    states = np.zeros((count, 13))
    states[0, :6] = np.array([radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0])
    states[0, 6:10] = np.array([1.0, 0.0, 0.0, 0.0])

    operational_states = []
    drag_areas_m2 = np.zeros(count)
    densities_kg_m3 = np.zeros(count)

    for index, time_s in enumerate(times_s):
        status, omega_rad_s = mission_status(time_s, scenario)
        operational_state = update_state(status)

        operational_states.append(operational_state)
        drag_areas_m2[index] = drag_area_from_state(operational_state)
        states[index, 10:13] = omega_rad_s

        altitude = np.linalg.norm(states[index, :3]) - earth_radius_km
        densities_kg_m3[index] = atmospheric_density_kg_m3(altitude)

        if index == count - 1:
            break

        states[index + 1, :6] = rk4_translation_step(
            states[index, :6], time_s, step_s, scenario
        )

        if operational_state == SpacecraftState.RECOVERED:
            states[index + 1, 6:10] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            states[index + 1, 6:10] = propagate_quaternion(
                states[index, 6:10], omega_rad_s, step_s
            )

    return SimulationResult(
        times_s=times_s,
        states=states,
        operational_states=operational_states,
        drag_areas_m2=drag_areas_m2,
        densities_kg_m3=densities_kg_m3,
    )


def semimajor_axis_km(states):
    """Compute osculating semimajor axis from Cartesian states."""
    radius = np.linalg.norm(states[:, :3], axis=1)
    speed_squared = np.sum(states[:, 3:6] ** 2, axis=1)
    specific_energy = 0.5 * speed_squared - mu_earth_km3_s2 / radius
    return -mu_earth_km3_s2 / (2.0 * specific_energy)


def print_transitions(label, result):
    """Print each operational-state transition."""
    print(f"\n{label} transitions:")
    previous = None
    for time_s, state in zip(result.times_s, result.operational_states):
        if state != previous:
            print(f"  t = {time_s / 3600.0:6.2f} h : {state.value}")
            previous = state


def main():
    environment_calibration = calibrate_environment_to_decay_rate()

    results = {
        "Nominal": simulate("nominal"),
        "Temporary tumble": simulate("temporary_tumble"),
        "No recovery": simulate("no_recovery"),
    }

    nominal = results["Nominal"]
    nominal_final_position = nominal.states[-1, :3]

    spacecraft_cases = {}
    for label, result in results.items():
        altitude = np.linalg.norm(result.states[:, :3], axis=1) - earth_radius_km
        final_displacement_km = np.linalg.norm(
            result.states[-1, :3] - nominal_final_position
        )
        spacecraft = create_spacecraft(label)

        spacecraft["orbit"].update(
            {
                "r": result.states[-1, :3],
                "v": result.states[-1, 3:6],
                "state_history_km_km_s_quat_rad_s": result.states,
                "altitude_history_km": altitude,
                "minimum_altitude_km": np.min(altitude),
                "final_altitude_km": altitude[-1],
                "final_displacement_from_nominal_km": final_displacement_km,
                "semimajor_axis_history_km": semimajor_axis_km(result.states),
            }
        )
        spacecraft["attitude"].update(
            {
                "q": result.states[-1, 6:10],
                "omega": result.states[-1, 10:13],
                "quaternion_history": result.states[:, 6:10],
                "angular_rate_history_rad_s": result.states[:, 10:13],
            }
        )
        spacecraft["vehicle"].update(
            {
                "mass": mass_kg,
                "cd": cd,
                "drag_area": result.drag_areas_m2[-1],
                "drag_area_history_m2": result.drag_areas_m2,
                "density_history_kg_m3": result.densities_kg_m3,
                "nominal_drag_area_m2": area_ram_m2,
                "tumbling_drag_area_m2": area_tumble_m2,
            }
        )
        spacecraft["operations"].update(
            {
                "state": result.operational_states[-1],
                "state_history": result.operational_states,
                "recovering": result.operational_states[-1] == SpacecraftState.RECOVERY,
                "recovered": result.operational_states[-1] == SpacecraftState.RECOVERED,
                "mission_lost": result.operational_states[-1]
                == SpacecraftState.MISSION_LOSS,
                "time_history_s": result.times_s,
                "time_history_h": result.times_s / 3600.0,
                "environment_calibration": environment_calibration,
            }
        )
        spacecraft_cases[label] = spacecraft

    print("Challenge 3, Part 2")
    print("-------------------")
    print(
        "Calibrated density scale: "
        f"{environment_calibration['density_scale']:.6f}"
    )
    print(
        "Reference decay rate:     "
        f"{environment_calibration['calibrated_decay_rate_km_per_day']:.3f} km/day"
    )
    print(f"Propagation duration: {nominal.times_s[-1] / 86400.0:.1f} days")
    print(f"Nominal area:         {area_ram_m2:.2f} m^2")
    print(f"Tumbling area:        {area_tumble_m2:.2f} m^2")

    for label, result in results.items():
        altitude = np.linalg.norm(result.states[:, :3], axis=1) - earth_radius_km
        final_displacement_km = np.linalg.norm(
            result.states[-1, :3] - nominal_final_position
        )
        print(
            f"{label:18s}: final altitude = {altitude[-1]:9.3f} km, "
            f"minimum altitude = {np.min(altitude):9.3f} km, "
            f"final separation from nominal = {final_displacement_km:9.3f} km"
        )
        print_transitions(label, result)

    plt.figure(figsize=(10, 6))
    for label, result in results.items():
        altitude = np.linalg.norm(result.states[:, :3], axis=1) - earth_radius_km
        plt.plot(result.times_s / 86400.0, altitude, label=label)
    plt.xlabel("Time [days]")
    plt.ylabel("Altitude [km]")
    plt.title("Challenge 3 Part 2: State-Dependent Orbital Decay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    for label, result in results.items():
        plt.plot(
            result.times_s / 86400.0,
            semimajor_axis_km(result.states),
            label=label,
        )
    plt.xlabel("Time [days]")
    plt.ylabel("Osculating semimajor axis [km]")
    plt.title("Semimajor-Axis Evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    for label, result in results.items():
        plt.step(
            result.times_s / 3600.0,
            result.drag_areas_m2,
            where="post",
            label=label,
        )
    plt.xlabel("Time [h]")
    plt.ylabel("Effective drag area [m^2]")
    plt.title("State-Selected Drag Area")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    for label, result in results.items():
        omega_deg_s = np.linalg.norm(result.states[:, 10:13], axis=1) * RAD2DEG
        plt.plot(result.times_s / 3600.0, omega_deg_s, label=label)
    plt.xlabel("Time [h]")
    plt.ylabel("Angular-rate magnitude [deg/s]")
    plt.title("Angular-Rate Histories")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    temporary = results["Temporary tumble"]
    plt.figure(figsize=(10, 6))
    for component in range(4):
        plt.plot(
            temporary.times_s / 3600.0,
            temporary.states[:, 6 + component],
            label=f"q{component}",
        )
    plt.xlabel("Time [h]")
    plt.ylabel("Quaternion component")
    plt.title("Temporary-Tumble Quaternion History")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    return spacecraft_cases


if __name__ == "__main__":
    main()
