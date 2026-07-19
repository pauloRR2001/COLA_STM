"""Challenge 1, Part B: intra-plane satellite slotting by differential mean motion.

A set of co-planar spacecraft begins co-located on the 520 km operational
orbit.  Each spacecraft is assigned a temporary circular phasing orbit so that
its accumulated mean-anomaly offset reaches an evenly spaced target slot after
a common phasing interval.  At the end of phasing, each spacecraft returns to
the operational altitude.

This is an analytical circular-orbit MVP.  It captures the dominant phasing
mechanism, n = sqrt(mu/a^3), without numerical force-model integration.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from constants import earth_radius_km, mu_earth_km3_s2, seconds_per_day, target_altitude_km


NUMBER_OF_SATELLITES = 4
PHASING_DURATION_DAYS = 7.0
TIME_STEP_SECONDS = 300.0


def mean_motion_rad_s(semi_major_axis_km: float | np.ndarray) -> float | np.ndarray:
    """Circular-orbit mean motion."""
    return np.sqrt(mu_earth_km3_s2 / np.asarray(semi_major_axis_km) ** 3)


def circular_speed_km_s(radius_km: float) -> float:
    return math.sqrt(mu_earth_km3_s2 / radius_km)


def round_trip_hohmann_delta_v_mps(radius_1_km: float, radius_2_km: float) -> float:
    """Delta-v for a Hohmann transfer from orbit 1 to orbit 2 and back.

    The return transfer has the same total impulsive cost as the outbound
    transfer, so the round-trip cost is twice the one-way two-impulse cost.
    """
    if math.isclose(radius_1_km, radius_2_km, rel_tol=0.0, abs_tol=1.0e-12):
        return 0.0

    transfer_axis_km = 0.5 * (radius_1_km + radius_2_km)
    v1 = circular_speed_km_s(radius_1_km)
    v2 = circular_speed_km_s(radius_2_km)
    vt1 = math.sqrt(mu_earth_km3_s2 * (2.0 / radius_1_km - 1.0 / transfer_axis_km))
    vt2 = math.sqrt(mu_earth_km3_s2 * (2.0 / radius_2_km - 1.0 / transfer_axis_km))
    one_way_km_s = abs(vt1 - v1) + abs(v2 - vt2)
    return 2.0 * one_way_km_s * 1000.0


def wrap_degrees(angle_deg: np.ndarray | float) -> np.ndarray | float:
    return np.mod(angle_deg, 360.0)


def main() -> None:
    operational_radius_km = earth_radius_km + target_altitude_km
    operational_mean_motion_rad_s = float(mean_motion_rad_s(operational_radius_km))
    phasing_duration_seconds = PHASING_DURATION_DAYS * seconds_per_day

    satellite_indices = np.arange(NUMBER_OF_SATELLITES)
    target_offsets_rad = 2.0 * np.pi * satellite_indices / NUMBER_OF_SATELLITES
    target_offsets_deg = np.degrees(target_offsets_rad)

    # Each spacecraft starts at the same phase.  A lower temporary orbit has a
    # larger mean motion and therefore advances in phase relative to satellite 0.
    required_delta_n_rad_s = target_offsets_rad / phasing_duration_seconds
    phasing_mean_motion_rad_s = operational_mean_motion_rad_s + required_delta_n_rad_s
    phasing_radius_km = (
        mu_earth_km3_s2 / phasing_mean_motion_rad_s**2
    ) ** (1.0 / 3.0)
    phasing_altitude_km = phasing_radius_km - earth_radius_km

    round_trip_delta_v_mps = np.array(
        [
            round_trip_hohmann_delta_v_mps(operational_radius_km, radius)
            for radius in phasing_radius_km
        ]
    )

    total_duration_seconds = phasing_duration_seconds + 2.0 * seconds_per_day
    times_seconds = np.arange(
        0.0, total_duration_seconds + TIME_STEP_SECONDS, TIME_STEP_SECONDS
    )
    times_days = times_seconds / seconds_per_day

    mean_anomaly_rad = np.zeros((NUMBER_OF_SATELLITES, len(times_seconds)))
    altitude_history_km = np.zeros_like(mean_anomaly_rad)

    for sat_index in satellite_indices:
        during_phasing = times_seconds <= phasing_duration_seconds
        after_phasing = ~during_phasing

        mean_anomaly_rad[sat_index, during_phasing] = (
            phasing_mean_motion_rad_s[sat_index] * times_seconds[during_phasing]
        )
        phase_at_end = (
            phasing_mean_motion_rad_s[sat_index] * phasing_duration_seconds
        )
        mean_anomaly_rad[sat_index, after_phasing] = (
            phase_at_end
            + operational_mean_motion_rad_s
            * (times_seconds[after_phasing] - phasing_duration_seconds)
        )

        altitude_history_km[sat_index, during_phasing] = phasing_altitude_km[sat_index]
        altitude_history_km[sat_index, after_phasing] = target_altitude_km

    relative_phase_deg = wrap_degrees(
        np.degrees(mean_anomaly_rad - mean_anomaly_rad[0:1, :])
    )
    final_index = int(np.argmin(np.abs(times_seconds - phasing_duration_seconds)))
    final_offsets_deg = relative_phase_deg[:, final_index]
    slot_error_deg = (
        (final_offsets_deg - target_offsets_deg + 180.0) % 360.0 - 180.0
    )

    print("Challenge 1 Part B — Intra-plane slotting")
    print(f"Operational altitude:  {target_altitude_km:.3f} km")
    print(f"Phasing duration:      {PHASING_DURATION_DAYS:.3f} days")
    print(f"Number of satellites: {NUMBER_OF_SATELLITES}")
    print()
    print(
        "Sat  Target slot [deg]  Phasing altitude [km]  "
        "Round-trip dV [m/s]  Final error [deg]"
    )
    for sat_index in satellite_indices:
        print(
            f"{sat_index:3d}"
            f"  {target_offsets_deg[sat_index]:17.6f}"
            f"  {phasing_altitude_km[sat_index]:21.6f}"
            f"  {round_trip_delta_v_mps[sat_index]:20.6f}"
            f"  {slot_error_deg[sat_index]:17.9f}"
        )

    print()
    print(f"Maximum absolute slot error: {np.max(np.abs(slot_error_deg)):.9e} deg")
    print(f"Total constellation phasing dV: {np.sum(round_trip_delta_v_mps):.6f} m/s")

    plt.figure()
    for sat_index in satellite_indices:
        plt.plot(
            times_days,
            relative_phase_deg[sat_index],
            label=f"Satellite {sat_index}",
        )
    for target in target_offsets_deg:
        plt.axhline(target, linewidth=0.8, linestyle="--")
    plt.axvline(PHASING_DURATION_DAYS, linewidth=1.0)
    plt.xlabel("Time [days]")
    plt.ylabel("Relative mean anomaly [deg]")
    plt.title("Intra-plane slot acquisition")
    plt.ylim(0.0, 360.0)
    plt.grid(True)
    plt.legend()

    plt.figure()
    for sat_index in satellite_indices:
        plt.step(
            times_days,
            altitude_history_km[sat_index],
            where="post",
            label=f"Satellite {sat_index}",
        )
    plt.axvline(PHASING_DURATION_DAYS, linewidth=1.0)
    plt.xlabel("Time [days]")
    plt.ylabel("Commanded circular-orbit altitude [km]")
    plt.title("Temporary phasing-orbit schedule")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.bar([f"Sat {index}" for index in satellite_indices], round_trip_delta_v_mps)
    plt.xlabel("Spacecraft")
    plt.ylabel("Round-trip phasing delta-v [m/s]")
    plt.title("Per-spacecraft slotting cost")
    plt.grid(True, axis="y")

    # Final constellation geometry in the common operational plane.
    final_theta_rad = np.radians(final_offsets_deg)
    x_km = operational_radius_km * np.cos(final_theta_rad)
    y_km = operational_radius_km * np.sin(final_theta_rad)
    theta_circle = np.linspace(0.0, 2.0 * np.pi, 500)

    plt.figure()
    plt.plot(
        operational_radius_km * np.cos(theta_circle),
        operational_radius_km * np.sin(theta_circle),
    )
    plt.scatter(x_km, y_km)
    for sat_index, (x_value, y_value) in enumerate(zip(x_km, y_km)):
        plt.annotate(f"Sat {sat_index}", (x_value, y_value))
    plt.xlabel("In-plane x [km]")
    plt.ylabel("In-plane y [km]")
    plt.title("Final evenly spaced constellation slots")
    plt.axis("equal")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
