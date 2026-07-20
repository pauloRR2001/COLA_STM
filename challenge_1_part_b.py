"""Challenge 1, Part B: Orekit finite-thrust intra-plane constellation slotting."""

import matplotlib.pyplot as plt
import numpy as np

from constants import area_ram_m2, cd, earth_radius_km, inclination_deg, isp_s, mass_kg, mu_earth_km3_s2, target_altitude_km, thrust_n
from functions import calibrate_environment_to_decay_rate, create_conjunction, propagate_schedule

NUMBER_OF_SATELLITES = 4
PHASING_DURATION_DAYS = 7.0
OUTPUT_STEP_SECONDS = 900.0


def circular_state():
    radius = earth_radius_km + target_altitude_km
    speed = np.sqrt(mu_earth_km3_s2 / radius)
    inc = np.radians(inclination_deg)
    return np.array([radius, 0.0, 0.0, 0.0, speed * np.cos(inc), speed * np.sin(inc)])


def wrap_degrees(angle):
    return np.mod(angle, 360.0)


def wrap_signed_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def orbital_phase_deg(states):
    """Argument of latitude in the fixed initial orbital plane."""
    inc = np.radians(inclination_deg)
    p = states[:, :3]
    x = p[:, 0]
    y_plane = p[:, 1] * np.cos(inc) + p[:, 2] * np.sin(inc)
    return np.degrees(np.unwrap(np.arctan2(y_plane, x)))


def phasing_delta_v_guess(target_offset_rad, duration_s):
    """First-order seed only; Orekit performs all trajectory propagation."""
    radius = earth_radius_km + target_altitude_km
    n = np.sqrt(mu_earth_km3_s2 / radius**3)
    delta_n = target_offset_rad / duration_s
    delta_a = -(2.0 / 3.0) * radius * delta_n / n
    speed = np.sqrt(mu_earth_km3_s2 / radius)
    return abs(0.5 * speed * delta_a / radius) * 1000.0


def main():
    calibration = calibrate_environment_to_decay_rate()
    common = {"area_m2": area_ram_m2, "cd": cd,
              "rho0_kg_m3": 3.5e-12 * calibration["density_scale"],
              "reference_altitude_km": 400.0, "scale_height_km": 58.0}
    duration_s = PHASING_DURATION_DAYS * 86400.0
    initial = circular_state()
    target_offsets_deg = 360.0 * np.arange(NUMBER_OF_SATELLITES) / NUMBER_OF_SATELLITES

    nominal_times, nominal_states, _ = propagate_schedule(
        initial, [{**common, "duration_s": duration_s}], OUTPUT_STEP_SECONDS)
    nominal_phase = orbital_phase_deg(nominal_states)

    histories = []
    conjunctions = [create_conjunction() for _ in range(NUMBER_OF_SATELLITES)]
    for index, target_deg in enumerate(target_offsets_deg):
        if index == 0:
            times, states = nominal_times, nominal_states
            burn_duration = 0.0
        else:
            signed_target = wrap_signed_degrees(target_deg)
            outbound = "RETROGRADE" if signed_target > 0.0 else "PROGRADE"
            inbound = "PROGRADE" if signed_target > 0.0 else "RETROGRADE"

            def propagate_candidate(candidate_duration):
                coast_duration = duration_s - 2.0 * candidate_duration
                segments = [
                    {**common, "duration_s": candidate_duration, "thrust_n": thrust_n,
                     "isp_s": isp_s, "thrust_direction": outbound},
                    {**common, "duration_s": coast_duration},
                    {**common, "duration_s": candidate_duration, "thrust_n": thrust_n,
                     "isp_s": isp_s, "thrust_direction": inbound},
                ]
                return propagate_schedule(initial, segments, OUTPUT_STEP_SECONDS)

            low = 0.0
            high = 0.499 * duration_s
            for _ in range(28):
                trial = 0.5 * (low + high)
                candidate_times, candidate_states, _ = propagate_candidate(trial)
                candidate_phase = orbital_phase_deg(candidate_states)
                nominal_at_candidate = np.interp(candidate_times, nominal_times, nominal_phase)
                achieved_signed = wrap_signed_degrees(
                    candidate_phase[-1] - nominal_at_candidate[-1]
                )
                if signed_target > 0.0:
                    if achieved_signed < signed_target:
                        low = trial
                    else:
                        high = trial
                else:
                    if achieved_signed > signed_target:
                        low = trial
                    else:
                        high = trial

            burn_duration = 0.5 * (low + high)
            times, states, _ = propagate_candidate(burn_duration)
        phase = orbital_phase_deg(states)
        # Segment boundaries can add or remove one output sample relative to the
        # single-segment nominal propagation. Compare the trajectories on the
        # candidate spacecraft's own time grid instead of assuming identical
        # array lengths.
        nominal_phase_on_times = np.interp(times, nominal_times, nominal_phase)
        relative = wrap_degrees(phase - nominal_phase_on_times)
        histories.append((times, states, relative, burn_duration))
        conjunctions[index]["primary"]["orbit"].update({"state_history": states, "time_history_s": times})
        conjunctions[index]["maneuver"].update({"epoch": 0.0, "magnitude": thrust_n,
            "duration_s": burn_duration, "executed": burn_duration > 0.0})

    print("Challenge 1, Part B — Orekit")
    print(" satellite | target slot [deg] | achieved slot [deg] | HET burn per arc [h]")
    for index, (target, history) in enumerate(zip(target_offsets_deg, histories)):
        achieved = history[2][-1]
        print(f" {index:9d} | {target:17.3f} | {achieved:19.3f} | {history[3] / 3600.0:18.3f}")

    plt.figure(figsize=(10, 6))
    for index, (times, _, relative, _) in enumerate(histories):
        plt.plot(times / 86400.0, relative, label=f"Satellite {index}")
    for target in target_offsets_deg:
        plt.axhline(target, linestyle="--", linewidth=0.7)
    plt.xlabel("Time [days]"); plt.ylabel("Relative orbital phase [deg]")
    plt.title("Orekit Finite-Thrust In-Plane Slotting"); plt.grid(True); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(7, 7))
    achieved = np.radians([h[2][-1] for h in histories])
    plt.scatter(np.cos(achieved), np.sin(achieved))
    for index, angle in enumerate(achieved):
        plt.text(1.05 * np.cos(angle), 1.05 * np.sin(angle), f"SC {index}", ha="center")
    circle = np.linspace(0.0, 2.0 * np.pi, 300)
    plt.plot(np.cos(circle), np.sin(circle)); plt.axis("equal"); plt.grid(True)
    plt.title("Final Orekit Constellation Geometry"); plt.tight_layout(); plt.show()
    return conjunctions


if __name__ == "__main__":
    main()
