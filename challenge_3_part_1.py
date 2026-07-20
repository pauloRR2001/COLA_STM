"""Challenge 3, Part 1: attitude-state and state-machine demonstration.

This script demonstrates the operational sequence

    NOMINAL -> TUMBLING -> RECOVERY -> RECOVERED

without modeling ADCS hardware.  Quaternion and angular-rate histories are
prescribed kinematically, while ``functions.state_machine.update_state``
determines the current operational mode from a status dictionary.
"""

import matplotlib.pyplot as plt
import numpy as np

from constants import area_ram_m2, area_tumble_m2
from functions import *







def main():
    environment_calibration = calibrate_environment_to_decay_rate()
    spacecraft = create_spacecraft("Challenge 3 Part 1 Spacecraft")

    step_s = 10.0
    duration_h = 8.0
    anomaly_time_h = 1.0
    recovery_start_h = 4.0
    recovery_complete_h = 6.0

    times_s = np.arange(0.0, duration_h * 3600.0 + step_s, step_s)
    times_h = times_s / 3600.0

    quaternions = np.zeros((len(times_s), 4))
    angular_rates_rad_s = np.zeros((len(times_s), 3))
    pointing_error_deg = np.zeros(len(times_s))
    drag_area_m2 = np.zeros(len(times_s))
    states = []

    quaternions[0] = np.array([1.0, 0.0, 0.0, 0.0])

    tumble_rate_rad_s = np.array([4.0, -2.0, 6.0]) * DEG2RAD
    recovery_time_constant_s = 1200.0

    for index, time_h in enumerate(times_h):
        recovering = recovery_start_h <= time_h < recovery_complete_h
        recovered = time_h >= recovery_complete_h

        if time_h < anomaly_time_h:
            omega_rad_s = np.zeros(3)
        elif time_h < recovery_start_h:
            omega_rad_s = tumble_rate_rad_s
        elif time_h < recovery_complete_h:
            elapsed_s = (time_h - recovery_start_h) * 3600.0
            omega_rad_s = tumble_rate_rad_s * np.exp(
                -elapsed_s / recovery_time_constant_s
            )
        else:
            omega_rad_s = np.zeros(3)

        status = {
            "omega_mag": np.linalg.norm(omega_rad_s) * RAD2DEG,
            "recovering": recovering,
            "recovered": recovered,
            "mission_lost": False,
        }
        state = update_state(status)
        states.append(state)
        angular_rates_rad_s[index] = omega_rad_s

        if state in (SpacecraftState.TUMBLING, SpacecraftState.RECOVERY):
            drag_area_m2[index] = area_tumble_m2
        else:
            drag_area_m2[index] = area_ram_m2

        if index > 0:
            if state == SpacecraftState.RECOVERED:
                quaternions[index] = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                quaternions[index] = propagate_quaternion(
                    quaternions[index - 1], omega_rad_s, step_s
                )

        pointing_error_deg[index] = quaternion_error_angle_deg(quaternions[index])

    state_codes = np.array([list(SpacecraftState).index(state) for state in states])

    spacecraft["attitude"].update(
        {
            "q": quaternions[-1],
            "omega": angular_rates_rad_s[-1],
            "quaternion_history": quaternions,
            "angular_rate_history_rad_s": angular_rates_rad_s,
            "pointing_error_history_deg": pointing_error_deg,
        }
    )
    spacecraft["vehicle"].update(
        {
            "drag_area": drag_area_m2[-1],
            "drag_area_history_m2": drag_area_m2,
            "nominal_drag_area_m2": area_ram_m2,
            "tumbling_drag_area_m2": area_tumble_m2,
        }
    )
    spacecraft["operations"].update(
        {
            "state": states[-1],
            "state_history": states,
            "state_code_history": state_codes,
            "recovering": states[-1] == SpacecraftState.RECOVERY,
            "recovered": states[-1] == SpacecraftState.RECOVERED,
            "mission_lost": states[-1] == SpacecraftState.MISSION_LOSS,
            "anomaly_time_h": anomaly_time_h,
            "recovery_start_h": recovery_start_h,
            "recovery_complete_h": recovery_complete_h,
            "time_history_s": times_s,
            "time_history_h": times_h,
            "environment_calibration": environment_calibration,
        }
    )

    print("Challenge 3, Part 1")
    print("-------------------")
    print(f"Tumble angular-rate magnitude: {np.linalg.norm(tumble_rate_rad_s) * RAD2DEG:.2f} deg/s")
    print(f"Nominal drag area:             {area_ram_m2:.2f} m^2")
    print(f"Tumbling drag area:            {area_tumble_m2:.2f} m^2")
    print(f"Drag-area multiplier:          {area_tumble_m2 / area_ram_m2:.2f}")
    print(
        "Calibrated density scale:      "
        f"{environment_calibration['density_scale']:.6f}"
    )
    print(
        "Reference decay rate:          "
        f"{environment_calibration['calibrated_decay_rate_km_per_day']:.3f} km/day"
    )
    print("\nState transitions:")

    previous_state = None
    for time_h, state in zip(times_h, states):
        if state != previous_state:
            print(f"  t = {time_h:5.2f} h : {state.value}")
            previous_state = state

    plt.figure(figsize=(10, 6))
    for component in range(4):
        plt.plot(times_h, quaternions[:, component], label=f"q{component}")
    plt.xlabel("Time [h]")
    plt.ylabel("Quaternion component")
    plt.title("Challenge 3 Part 1: Attitude Quaternion")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    angular_rates_deg_s = angular_rates_rad_s * RAD2DEG
    labels = ("omega_x", "omega_y", "omega_z")
    for component, label in enumerate(labels):
        plt.plot(times_h, angular_rates_deg_s[:, component], label=label)
    plt.plot(
        times_h,
        np.linalg.norm(angular_rates_deg_s, axis=1),
        "--",
        label="|omega|",
    )
    plt.xlabel("Time [h]")
    plt.ylabel("Angular rate [deg/s]")
    plt.title("Angular-Rate History")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.step(times_h, state_codes, where="post")
    plt.yticks(range(len(SpacecraftState)), [state.value for state in SpacecraftState])
    plt.xlabel("Time [h]")
    plt.ylabel("Operational state")
    plt.title("Spacecraft State-Machine Timeline")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(times_h, pointing_error_deg, label="Pointing error")
    plt.xlabel("Time [h]")
    plt.ylabel("Principal pointing error [deg]")
    plt.title("Attitude Error Relative to Nominal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.step(times_h, drag_area_m2, where="post")
    plt.xlabel("Time [h]")
    plt.ylabel("Effective drag area [m^2]")
    plt.title("State-Dependent Drag Configuration")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    return spacecraft


if __name__ == "__main__":
    main()
