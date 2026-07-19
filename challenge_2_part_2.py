"""Challenge 2, Part 2: differential-drag collision avoidance.

The primary spacecraft normally flies at the ram-facing area.  For avoidance it
switches to the maximum drag area for a fixed interval before TCA, then returns
to the nominal area.  The script searches the high-drag lead time until the
estimated collision probability falls below the configured threshold.
"""

import matplotlib.pyplot as plt
import numpy as np

from constants import (
    area_max_m2,
    area_ram_m2,
    cd,
    cola_altitude_km,
    cola_lead_time_days,
    cola_step_seconds,
    collision_probability_threshold,
    drag_reference_altitude_km,
    drag_reference_density_kg_m3,
    drag_scale_height_km,
    earth_radius_km,
    earth_rotation_rate_rad_s,
    hard_body_radius_km,
    high_drag_duration_hours,
    inclination_deg,
    lead_time_step_hours,
    mass_kg,
    maximum_lead_time_hours,
    minimum_lead_time_hours,
    mu_earth_km3_s2,
    primary_position_sigma_km,
    primary_velocity_sigma_km_s,
    secondary_position_sigma_km,
    secondary_velocity_sigma_km_s,
    seconds_per_day,
    target_miss_rtn_km,
    target_relative_velocity_rtn_km_s,
)
from functions.cola import (
    assess_conjunction,
    combined_acceleration,
    create_synthetic_encounter,
    propagate,
    rtn_basis,
    two_body_acceleration,
    write_json,
)


def covariance(position_sigma_km, velocity_sigma_km_s):
    """Create a simple diagonal Cartesian state covariance."""
    return np.diag(
        [position_sigma_km**2] * 3 + [velocity_sigma_km_s**2] * 3
    )


def atmospheric_density_kg_m3(altitude_km):
    """Simple exponential density model referenced to the COLA altitude."""
    return drag_reference_density_kg_m3 * np.exp(
        -(altitude_km - drag_reference_altitude_km) / drag_scale_height_km
    )


def drag_model(area_schedule):
    """Return an atmospheric-drag acceleration model.

    ``area_schedule`` is an iterable of ``(start, stop, area_m2)`` intervals.
    Outside those intervals the spacecraft uses ``area_ram_m2``.
    """

    omega_earth = np.array([0.0, 0.0, earth_rotation_rate_rad_s])

    def model(state, time):
        r_km = state[:3]
        v_km_s = state[3:]
        altitude_km = np.linalg.norm(r_km) - earth_radius_km
        density = atmospheric_density_kg_m3(altitude_km)

        area_m2 = area_ram_m2
        for start, stop, scheduled_area in area_schedule:
            if start <= time < stop:
                area_m2 = scheduled_area
                break

        # Atmosphere co-rotates with Earth.  Convert its velocity to km/s.
        atmosphere_velocity_km_s = np.cross(omega_earth, r_km)
        relative_velocity_km_s = v_km_s - atmosphere_velocity_km_s
        relative_speed_m_s = np.linalg.norm(relative_velocity_km_s) * 1000.0

        if relative_speed_m_s == 0.0:
            return np.zeros(3)

        # Drag magnitude is computed in SI, then converted from m/s^2 to km/s^2.
        acceleration_m_s2 = (
            -0.5
            * density
            * cd
            * area_m2
            / mass_kg
            * relative_speed_m_s
            * (relative_velocity_km_s * 1000.0)
        )
        return acceleration_m_s2 / 1000.0

    return model


def main():
    gravity = two_body_acceleration(mu_earth_km3_s2)

    primary_tca, secondary_tca = create_synthetic_encounter(
        mu_earth_km3_s2,
        earth_radius_km,
        cola_altitude_km,
        inclination_deg,
        target_miss_rtn_km,
        target_relative_velocity_rtn_km_s,
    )

    primary_covariance = covariance(
        primary_position_sigma_km, primary_velocity_sigma_km_s
    )
    secondary_covariance = covariance(
        secondary_position_sigma_km, secondary_velocity_sigma_km_s
    )

    nominal_tca_seconds = cola_lead_time_days * seconds_per_day

    # Build the epoch states by integrating the designed TCA geometry backward.
    primary_backward = propagate(
        primary_tca,
        primary_covariance,
        -nominal_tca_seconds,
        -cola_step_seconds,
        gravity,
        mu_earth_km3_s2,
    )
    secondary_backward = propagate(
        secondary_tca,
        secondary_covariance,
        -nominal_tca_seconds,
        -cola_step_seconds,
        gravity,
        mu_earth_km3_s2,
    )
    primary_initial = primary_backward.states[-1]
    secondary_initial = secondary_backward.states[-1]

    # Continue beyond TCA to show that the vehicle returns to nominal area.
    end_time_seconds = nominal_tca_seconds + 12.0 * 3600.0

    nominal_primary_model = combined_acceleration(gravity, drag_model(()))
    primary_nominal = propagate(
        primary_initial,
        primary_covariance,
        end_time_seconds,
        cola_step_seconds,
        nominal_primary_model,
        mu_earth_km3_s2,
    )

    # The secondary is kept ballistic in this intentionally simple MVP.
    secondary = propagate(
        secondary_initial,
        secondary_covariance,
        end_time_seconds,
        cola_step_seconds,
        gravity,
        mu_earth_km3_s2,
    )
    nominal = assess_conjunction(primary_nominal, secondary, hard_body_radius_km)

    high_drag_duration_seconds = high_drag_duration_hours * 3600.0
    tested_leads = []
    tested_probabilities = []
    tested_miss_distances_m = []
    selected = None

    for lead_hours in np.arange(
        minimum_lead_time_hours,
        maximum_lead_time_hours + lead_time_step_hours,
        lead_time_step_hours,
    ):
        high_drag_start = nominal.tca_seconds - lead_hours * 3600.0
        high_drag_stop = high_drag_start + high_drag_duration_seconds

        # Require the complete high-drag interval to occur before TCA.
        if high_drag_start < 0.0 or high_drag_stop >= nominal.tca_seconds:
            continue

        schedule = ((high_drag_start, high_drag_stop, area_max_m2),)
        maneuver_model = combined_acceleration(gravity, drag_model(schedule))
        primary_high_drag = propagate(
            primary_initial,
            primary_covariance,
            end_time_seconds,
            cola_step_seconds,
            maneuver_model,
            mu_earth_km3_s2,
        )
        result = assess_conjunction(primary_high_drag, secondary, hard_body_radius_km)

        tested_leads.append(float(lead_hours))
        tested_probabilities.append(result.collision_probability)
        tested_miss_distances_m.append(result.miss_distance_km * 1000.0)

        if result.collision_probability < collision_probability_threshold:
            selected = (lead_hours, schedule, primary_high_drag, result)
            break

    if not tested_probabilities:
        raise RuntimeError(
            "No valid high-drag interval exists. Increase the maximum lead time "
            "or reduce high_drag_duration_hours."
        )

    if selected is None:
        best = int(np.argmin(tested_probabilities))
        lead_hours = tested_leads[best]
        high_drag_start = nominal.tca_seconds - lead_hours * 3600.0
        schedule = (
            (
                high_drag_start,
                high_drag_start + high_drag_duration_seconds,
                area_max_m2,
            ),
        )
        maneuver_model = combined_acceleration(gravity, drag_model(schedule))
        primary_high_drag = propagate(
            primary_initial,
            primary_covariance,
            end_time_seconds,
            cola_step_seconds,
            maneuver_model,
            mu_earth_km3_s2,
        )
        selected = (
            lead_hours,
            schedule,
            primary_high_drag,
            assess_conjunction(primary_high_drag, secondary, hard_body_radius_km),
        )

    lead_hours, schedule, primary_high_drag, high_drag_result = selected
    high_drag_start, high_drag_stop, _ = schedule[0]

    displacement_at_nominal_tca_eci = (
        primary_high_drag.states[nominal.index, :3]
        - primary_nominal.states[nominal.index, :3]
    )
    displacement_at_nominal_tca_rtn = (
        rtn_basis(primary_nominal.states[nominal.index]).T
        @ displacement_at_nominal_tca_eci
    )
    along_track_displacement_m = displacement_at_nominal_tca_rtn[1] * 1000.0

    nominal_speed_km_s = np.linalg.norm(primary_nominal.states[nominal.index, 3:])
    equivalent_timing_shift_seconds = (
        displacement_at_nominal_tca_rtn[1] / nominal_speed_km_s
    )

    payload = {
        "MESSAGE_ID": "ARTIFICIAL-COLA-DIFFERENTIAL-DRAG-001",
        "REFERENCE_FRAME": "ECI",
        "COVARIANCE_FRAME": "ECI",
        "TCA_SECONDS_FROM_EPOCH": nominal.tca_seconds,
        "MISS_DISTANCE_M": nominal.miss_distance_km * 1000.0,
        "COLLISION_PROBABILITY": nominal.collision_probability,
        "COLLISION_PROBABILITY_THRESHOLD": collision_probability_threshold,
        "RELATIVE_POSITION_RTN_M": (
            nominal.relative_position_rtn_km * 1000.0
        ).tolist(),
        "RELATIVE_VELOCITY_RTN_MPS": (
            nominal.relative_velocity_rtn_km_s * 1000.0
        ).tolist(),
        "HARD_BODY_RADIUS_M": hard_body_radius_km * 1000.0,
        "RECOMMENDED_MANEUVER": {
            "TYPE": "DIFFERENTIAL_DRAG",
            "NOMINAL_AREA_M2": area_ram_m2,
            "HIGH_DRAG_AREA_M2": area_max_m2,
            "HIGH_DRAG_LEAD_TIME_HOURS": float(lead_hours),
            "HIGH_DRAG_START_SECONDS_FROM_EPOCH": float(high_drag_start),
            "HIGH_DRAG_STOP_SECONDS_FROM_EPOCH": float(high_drag_stop),
            "HIGH_DRAG_DURATION_HOURS": high_drag_duration_hours,
            "RETURN_TO_NOMINAL_AREA_AFTER_INTERVAL": True,
            "ALONG_TRACK_DISPLACEMENT_AT_NOMINAL_TCA_M": float(
                along_track_displacement_m
            ),
            "EQUIVALENT_TIMING_SHIFT_SECONDS": float(
                equivalent_timing_shift_seconds
            ),
            "POST_MANEUVER_TCA_SECONDS_FROM_EPOCH": high_drag_result.tca_seconds,
            "POST_MANEUVER_MISS_DISTANCE_M": (
                high_drag_result.miss_distance_km * 1000.0
            ),
            "POST_MANEUVER_COLLISION_PROBABILITY": (
                high_drag_result.collision_probability
            ),
        },
        "PRIMARY_COVARIANCE_AT_TCA": primary_nominal.covariances[
            nominal.index
        ].tolist(),
        "SECONDARY_COVARIANCE_AT_TCA": secondary.covariances[
            nominal.index
        ].tolist(),
    }
    write_json("artificial_cdm_differential_drag.json", payload)

    print(f"Initial Pc:                    {nominal.collision_probability:.6e}")
    print(f"Threshold:                     {collision_probability_threshold:.6e}")
    print(f"Selected high-drag lead time:  {lead_hours:.1f} h")
    print(f"High-drag duration:            {high_drag_duration_hours:.1f} h")
    print(f"Area change:                   {area_ram_m2:.2f} -> {area_max_m2:.2f} m^2")
    print(f"Initial miss distance:         {nominal.miss_distance_km * 1000.0:.3f} m")
    print(f"Post-drag Pc:                  {high_drag_result.collision_probability:.6e}")
    print(f"Post-drag miss distance:       {high_drag_result.miss_distance_km * 1000.0:.3f} m")
    print(f"Along-track displacement:      {along_track_displacement_m:.3f} m")
    print(f"Equivalent timing shift:       {equivalent_timing_shift_seconds:.3f} s")
    print("Artificial CDM written to artificial_cdm_differential_drag.json")

    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*primary_nominal.states[:, :3].T, label="Primary nominal drag")
    axis.plot(*primary_high_drag.states[:, :3].T, label="Primary high drag")
    axis.plot(*secondary.states[:, :3].T, label="Secondary")
    axis.set_xlabel("X [km]")
    axis.set_ylabel("Y [km]")
    axis.set_zlabel("Z [km]")
    axis.set_box_aspect((1, 1, 1))
    axis.legend()

    plt.figure()
    plt.plot(
        primary_nominal.times / 3600.0,
        nominal.distances_km * 1000.0,
        label="Nominal drag",
    )
    plt.plot(
        primary_high_drag.times / 3600.0,
        high_drag_result.distances_km * 1000.0,
        label="Differential drag",
    )
    plt.axvline(nominal.tca_seconds / 3600.0)
    plt.xlabel("Time from epoch [h]")
    plt.ylabel("Relative distance [m]")
    plt.legend()

    plt.figure()
    plt.semilogy(tested_leads, tested_probabilities, marker="o")
    plt.axhline(collision_probability_threshold)
    plt.xlabel("High-drag lead time [h]")
    plt.ylabel("Estimated collision probability")

    displacement_rtn = np.array(
        [
            rtn_basis(primary_nominal.states[index]).T
            @ (
                primary_high_drag.states[index, :3]
                - primary_nominal.states[index, :3]
            )
            for index in range(len(primary_nominal.times))
        ]
    )
    plt.figure()
    plt.plot(primary_nominal.times / 3600.0, displacement_rtn[:, 0] * 1000.0, label="R")
    plt.plot(primary_nominal.times / 3600.0, displacement_rtn[:, 1] * 1000.0, label="T")
    plt.plot(primary_nominal.times / 3600.0, displacement_rtn[:, 2] * 1000.0, label="N")
    plt.axvspan(high_drag_start / 3600.0, high_drag_stop / 3600.0, alpha=0.2)
    plt.xlabel("Time from epoch [h]")
    plt.ylabel("Differential-drag displacement [m]")
    plt.legend()

    plt.figure()
    area_history = np.full_like(primary_nominal.times, area_ram_m2, dtype=float)
    active = (primary_nominal.times >= high_drag_start) & (
        primary_nominal.times < high_drag_stop
    )
    area_history[active] = area_max_m2
    plt.step(primary_nominal.times / 3600.0, area_history, where="post")
    plt.xlabel("Time from epoch [h]")
    plt.ylabel("Effective drag area [m²]")

    # Spacecraft-to-spacecraft relative motion in the primary-centered RTN frame.
    relative_nominal_rtn = np.array(
        [
            rtn_basis(primary_nominal.states[index]).T
            @ (
                secondary.states[index, :3]
                - primary_nominal.states[index, :3]
            )
            for index in range(len(primary_nominal.times))
        ]
    )
    relative_high_drag_rtn = np.array(
        [
            rtn_basis(primary_high_drag.states[index]).T
            @ (
                secondary.states[index, :3]
                - primary_high_drag.states[index, :3]
            )
            for index in range(len(primary_high_drag.times))
        ]
    )

    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(
        relative_nominal_rtn[:, 0] * 1000.0,
        relative_nominal_rtn[:, 1] * 1000.0,
        relative_nominal_rtn[:, 2] * 1000.0,
        label="Nominal relative trajectory",
    )
    axis.plot(
        relative_high_drag_rtn[:, 0] * 1000.0,
        relative_high_drag_rtn[:, 1] * 1000.0,
        relative_high_drag_rtn[:, 2] * 1000.0,
        label="High-drag relative trajectory",
    )
    axis.scatter(0.0, 0.0, 0.0, marker="x", label="Primary spacecraft")
    axis.set_xlabel("Radial, R [m]")
    axis.set_ylabel("Along-track, T [m]")
    axis.set_zlabel("Cross-track, N [m]")
    axis.set_title("Secondary Motion in the Primary RTN Frame")
    axis.legend()

    plt.figure()
    time_to_nominal_tca_hours = (
        primary_nominal.times - nominal.tca_seconds
    ) / 3600.0
    plt.plot(
        time_to_nominal_tca_hours,
        relative_nominal_rtn[:, 0] * 1000.0,
        label="Nominal R",
    )
    plt.plot(
        time_to_nominal_tca_hours,
        relative_nominal_rtn[:, 1] * 1000.0,
        label="Nominal T",
    )
    plt.plot(
        time_to_nominal_tca_hours,
        relative_nominal_rtn[:, 2] * 1000.0,
        label="Nominal N",
    )
    plt.plot(
        time_to_nominal_tca_hours,
        relative_high_drag_rtn[:, 0] * 1000.0,
        linestyle="--",
        label="High-drag R",
    )
    plt.plot(
        time_to_nominal_tca_hours,
        relative_high_drag_rtn[:, 1] * 1000.0,
        linestyle="--",
        label="High-drag T",
    )
    plt.plot(
        time_to_nominal_tca_hours,
        relative_high_drag_rtn[:, 2] * 1000.0,
        linestyle="--",
        label="High-drag N",
    )
    plt.axvline(0.0)
    plt.xlabel("Time from nominal TCA [h]")
    plt.ylabel("Secondary relative position [m]")
    plt.title("Relative RTN Components")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
