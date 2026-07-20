"""Challenge 2, Part 1: synthetic CDM and low-thrust collision avoidance."""

import matplotlib.pyplot as plt
import numpy as np

from constants import (
    avoidance_burn_duration_seconds,
    cola_altitude_km,
    cola_lead_time_days,
    cola_step_seconds,
    collision_probability_threshold,
    earth_radius_km,
    hard_body_radius_km,
    inclination_deg,
    lead_time_step_hours,
    mass_kg,
    maximum_lead_time_hours,
    minimum_lead_time_hours,
    mu_earth_km3_s2,
    primary_position_sigma_km,
    primary_velocity_sigma_km_s,
    restore_delay_seconds,
    secondary_position_sigma_km,
    secondary_velocity_sigma_km_s,
    seconds_per_day,
    target_miss_rtn_km,
    target_relative_velocity_rtn_km_s,
    thrust_n,
)
from functions.cola import (
    assess_conjunction,
    assess_conjunction_at_time,
    combined_acceleration,
    create_synthetic_encounter,
    propagate,
    rtn_basis,
    two_body_acceleration,
    write_json,
)
from functions.models import create_conjunction


def thrust_model(burn_windows):
    windows = []
    for start, stop, direction in burn_windows:
        windows.append(
            (
                start,
                stop,
                thrust_n,
                "PROGRADE" if direction >= 0.0 else "RETROGRADE",
            )
        )
    return {"engine": "orekit", "thrust_windows": windows, "area_windows": []}


def covariance(position_sigma_km, velocity_sigma_km_s):
    return np.diag(
        [position_sigma_km**2] * 3 + [velocity_sigma_km_s**2] * 3
    )


def main():
    conjunction = create_conjunction()
    primary = conjunction["primary"]
    secondary_spacecraft = conjunction["secondary"]
    collision = conjunction["collision"]
    maneuver = conjunction["maneuver"]

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

    primary["orbit"].update(
        {
            "r": primary_initial[:3],
            "v": primary_initial[3:],
            "state_eci_km_km_s": primary_initial,
            "covariance": primary_covariance,
        }
    )
    primary["vehicle"].update(
        {
            "mass": mass_kg,
        }
    )

    secondary_spacecraft["orbit"].update(
        {
            "r": secondary_initial[:3],
            "v": secondary_initial[3:],
            "state_eci_km_km_s": secondary_initial,
            "covariance": secondary_covariance,
        }
    )

    end_time_seconds = (
        nominal_tca_seconds
        + restore_delay_seconds
        + avoidance_burn_duration_seconds
        + 6.0 * 3600.0
    )
    primary_nominal = propagate(
        primary_initial,
        primary_covariance,
        end_time_seconds,
        cola_step_seconds,
        gravity,
        mu_earth_km3_s2,
    )
    secondary = propagate(
        secondary_initial,
        secondary_covariance,
        end_time_seconds,
        cola_step_seconds,
        gravity,
        mu_earth_km3_s2,
    )
    nominal = assess_conjunction_at_time(
        primary_nominal,
        secondary,
        nominal_tca_seconds,
        hard_body_radius_km,
    )

    tested_leads = []
    tested_probabilities = []
    selected = None

    for lead_hours in np.arange(
        minimum_lead_time_hours,
        maximum_lead_time_hours + lead_time_step_hours,
        lead_time_step_hours,
    ):
        burn_start = nominal.tca_seconds - lead_hours * 3600.0
        burn_stop = burn_start + avoidance_burn_duration_seconds
        if burn_start < 0.0 or burn_stop >= nominal.tca_seconds:
            continue

        restore_start = nominal.tca_seconds + restore_delay_seconds
        restore_stop = restore_start + avoidance_burn_duration_seconds
        windows = (
            (burn_start, burn_stop, 1.0),
            (restore_start, restore_stop, -1.0),
        )
        model = combined_acceleration(gravity, thrust_model(windows))
        propagated = propagate(
            primary_initial,
            primary_covariance,
            end_time_seconds,
            cola_step_seconds,
            model,
            mu_earth_km3_s2,
        )
        result = assess_conjunction(propagated, secondary, hard_body_radius_km)
        tested_leads.append(float(lead_hours))
        tested_probabilities.append(result.collision_probability)

        if result.collision_probability < collision_probability_threshold:
            selected = (lead_hours, windows, propagated, result)
            break

    if selected is None:
        best = int(np.argmin(tested_probabilities))
        lead_hours = tested_leads[best]
        burn_start = nominal.tca_seconds - lead_hours * 3600.0
        burn_stop = burn_start + avoidance_burn_duration_seconds
        restore_start = nominal.tca_seconds + restore_delay_seconds
        windows = (
            (burn_start, burn_stop, 1.0),
            (restore_start, restore_start + avoidance_burn_duration_seconds, -1.0),
        )
        model = combined_acceleration(gravity, thrust_model(windows))
        propagated = propagate(
            primary_initial,
            primary_covariance,
            end_time_seconds,
            cola_step_seconds,
            model,
            mu_earth_km3_s2,
        )
        selected = (
            lead_hours,
            windows,
            propagated,
            assess_conjunction(propagated, secondary, hard_body_radius_km),
        )

    lead_hours, windows, primary_maneuvered, maneuvered = selected
    delta_v_mps = thrust_n / mass_kg * avoidance_burn_duration_seconds

    primary["orbit"].update(
        {
            "nominal": primary_nominal,
            "maneuvered": primary_maneuvered,
            "state_at_tca_eci_km_km_s": primary_nominal.states[nominal.index],
            "maneuvered_state_at_tca_eci_km_km_s": primary_maneuvered.states[
                maneuvered.index
            ],
        }
    )
    secondary_spacecraft["orbit"].update(
        {
            "propagation": secondary,
            "state_at_tca_eci_km_km_s": secondary.states[nominal.index],
        }
    )

    collision.update(
        {
            "tca": nominal.tca_seconds,
            "distance": nominal.miss_distance_km,
            "probability": nominal.collision_probability,
            "warning": nominal.collision_probability >= collision_probability_threshold,
            "collision": nominal.miss_distance_km <= hard_body_radius_km,
            "maneuver_required": (
                nominal.collision_probability >= collision_probability_threshold
            ),
            "threshold": collision_probability_threshold,
            "hard_body_radius_km": hard_body_radius_km,
            "relative_position_eci_km": nominal.relative_position_eci_km,
            "relative_velocity_eci_km_s": nominal.relative_velocity_eci_km_s,
            "relative_position_rtn_km": nominal.relative_position_rtn_km,
            "relative_velocity_rtn_km_s": nominal.relative_velocity_rtn_km_s,
            "combined_position_covariance_rtn_km2": (
                nominal.combined_position_covariance_rtn_km2
            ),
            "post_maneuver_tca": maneuvered.tca_seconds,
            "post_maneuver_distance": maneuvered.miss_distance_km,
            "post_maneuver_probability": maneuvered.collision_probability,
            "post_maneuver_relative_position_rtn_km": (
                maneuvered.relative_position_rtn_km
            ),
            "post_maneuver_relative_velocity_rtn_km_s": (
                maneuvered.relative_velocity_rtn_km_s
            ),
            "tested_lead_times_hours": tested_leads,
            "tested_probabilities": tested_probabilities,
        }
    )

    maneuver.update(
        {
            "epoch": float(windows[0][0]),
            "type": "CONTINUOUS_LOW_THRUST",
            "frame": "RTN",
            "direction": np.array([0.0, 1.0, 0.0]),
            "restore_direction": np.array([0.0, -1.0, 0.0]),
            "lead_time_hours": float(lead_hours),
            "duration_seconds": avoidance_burn_duration_seconds,
            "restore_delay_seconds_after_tca": restore_delay_seconds,
            "windows": windows,
            "thrust_n": thrust_n,
            "delta_v": delta_v_mps,
            "magnitude": delta_v_mps,
            "executed": False,
        }
    )

    payload = {
        "MESSAGE_ID": "ARTIFICIAL-COLA-LOW-THRUST-001",
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
            "TYPE": "CONTINUOUS_LOW_THRUST",
            "DIRECTION_RTN": [0.0, 1.0, 0.0],
            "LEAD_TIME_HOURS": float(lead_hours),
            "BURN_DURATION_SECONDS": avoidance_burn_duration_seconds,
            "DELTA_V_MPS": delta_v_mps,
            "RESTORE_DIRECTION_RTN": [0.0, -1.0, 0.0],
            "RESTORE_DELAY_SECONDS_AFTER_TCA": restore_delay_seconds,
            "POST_MANEUVER_MISS_DISTANCE_M": maneuvered.miss_distance_km * 1000.0,
            "POST_MANEUVER_COLLISION_PROBABILITY": maneuvered.collision_probability,
        },
        "PRIMARY_COVARIANCE_AT_TCA": primary_nominal.covariances[
            nominal.index
        ].tolist(),
        "SECONDARY_COVARIANCE_AT_TCA": secondary.covariances[
            nominal.index
        ].tolist(),
    }
    write_json("artificial_cdm_low_thrust.json", payload)

    print(f"Initial Pc:                 {nominal.collision_probability:.6e}")
    print(f"Threshold:                  {collision_probability_threshold:.6e}")
    print(f"Selected burn lead time:    {lead_hours:.1f} h")
    print(f"Burn duration:              {avoidance_burn_duration_seconds:.0f} s")
    print(f"Single-burn delta-v:        {delta_v_mps:.6f} m/s")
    print(f"Initial miss distance:      {nominal.miss_distance_km * 1000.0:.3f} m")
    print(f"Post-maneuver Pc:           {maneuvered.collision_probability:.6e}")
    print(f"Post-maneuver miss distance:{maneuvered.miss_distance_km * 1000.0:.3f} m")

    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*primary_nominal.states[:, :3].T, label="Primary nominal")
    axis.plot(*primary_maneuvered.states[:, :3].T, label="Primary maneuvered")
    axis.plot(*secondary.states[:, :3].T, label="Secondary")
    axis.set_xlabel("X [km]")
    axis.set_ylabel("Y [km]")
    axis.set_zlabel("Z [km]")
    axis.set_box_aspect((1, 1, 1))
    axis.legend()

    plt.figure()
    plt.plot(primary_nominal.times / 3600.0, nominal.distances_km * 1000.0, label="Nominal")
    plt.plot(primary_maneuvered.times / 3600.0, maneuvered.distances_km * 1000.0, label="Maneuvered")
    plt.axvline(nominal.tca_seconds / 3600.0)
    plt.xlabel("Time from epoch [h]")
    plt.ylabel("Relative distance [m]")
    plt.legend()

    plt.figure()
    plt.semilogy(tested_leads, tested_probabilities, marker="o")
    plt.axhline(collision_probability_threshold)
    plt.xlabel("Burn lead time [h]")
    plt.ylabel("Estimated collision probability")

    displacement_rtn = np.array(
        [
            rtn_basis(primary_nominal.states[index]).T
            @ (
                primary_maneuvered.states[index, :3]
                - primary_nominal.states[index, :3]
            )
            for index in range(len(primary_nominal.times))
        ]
    )
    plt.figure()
    plt.plot(primary_nominal.times / 3600.0, displacement_rtn[:, 0] * 1000.0, label="R")
    plt.plot(primary_nominal.times / 3600.0, displacement_rtn[:, 1] * 1000.0, label="T")
    plt.plot(primary_nominal.times / 3600.0, displacement_rtn[:, 2] * 1000.0, label="N")
    plt.xlabel("Time from epoch [h]")
    plt.ylabel("Maneuver displacement [m]")
    plt.legend()

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
    relative_maneuvered_rtn = np.array(
        [
            rtn_basis(primary_maneuvered.states[index]).T
            @ (
                secondary.states[index, :3]
                - primary_maneuvered.states[index, :3]
            )
            for index in range(len(primary_maneuvered.times))
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
        relative_maneuvered_rtn[:, 0] * 1000.0,
        relative_maneuvered_rtn[:, 1] * 1000.0,
        relative_maneuvered_rtn[:, 2] * 1000.0,
        label="Maneuvered relative trajectory",
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
        relative_maneuvered_rtn[:, 0] * 1000.0,
        linestyle="--",
        label="Maneuvered R",
    )
    plt.plot(
        time_to_nominal_tca_hours,
        relative_maneuvered_rtn[:, 1] * 1000.0,
        linestyle="--",
        label="Maneuvered T",
    )
    plt.plot(
        time_to_nominal_tca_hours,
        relative_maneuvered_rtn[:, 2] * 1000.0,
        linestyle="--",
        label="Maneuvered N",
    )
    plt.axvline(0.0)
    plt.xlabel("Time from nominal TCA [h]")
    plt.ylabel("Secondary relative position [m]")
    plt.title("Relative RTN Components")
    plt.legend()
    plt.show()

    return conjunction


if __name__ == "__main__":
    main()
