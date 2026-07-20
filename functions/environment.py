"""Environment and drag helper functions used by the challenge simulations."""

import numpy as np

from constants import (
    area_ram_m2,
    area_tumble_m2,
    cd,
    earth_radius_km,
    earth_rotation_rate_rad_s,
    initial_altitude_km,
    mass_kg,
    mu_earth_km3_s2,
    nominal_decay_rate_km_per_day,
    seconds_per_day,
)
from functions.state_machine import SpacecraftState

EARTH_ROTATION_RAD_S = earth_rotation_rate_rad_s
ATMOSPHERIC_DENSITY_SCALE = 1.0


def base_atmospheric_density_kg_m3(altitude_km):
    """Uncalibrated exponential atmosphere used as the model shape."""
    rho0 = 3.5e-12
    h0 = 400.0
    H = 58.0
    return rho0 * np.exp(np.clip(-(altitude_km - h0) / H, -50, 50))


def atmospheric_density_kg_m3(altitude_km):
    """Calibrated atmospheric density model.

    The exponential atmosphere supplies the altitude dependence.  The global
    scale factor is set by ``calibrate_environment_to_decay_rate`` so Challenge 3
    can reproduce the provided solar-maximum reference decay rate at 330 km.
    """
    return ATMOSPHERIC_DENSITY_SCALE * base_atmospheric_density_kg_m3(altitude_km)


def drag_area_from_state(state):
    """Select effective drag area from spacecraft operational state."""
    if state in (SpacecraftState.TUMBLING, SpacecraftState.RECOVERY):
        return area_tumble_m2
    return area_ram_m2


def circular_drag_decay_rate_km_per_day(
    altitude_km,
    density_kg_m3,
    area_m2=area_ram_m2,
    spacecraft_mass_kg=mass_kg,
    drag_coefficient=cd,
    mu_km3_s2=mu_earth_km3_s2,
    radius_body_km=earth_radius_km,
    earth_rotation_rad_s=EARTH_ROTATION_RAD_S,
):
    """Estimate circular-orbit altitude decay rate from drag.

    The estimate is consistent with the simulation drag law.  It assumes an
    initially circular prograde equatorial state so the atmosphere-relative speed
    is the inertial circular speed minus the rotating-atmosphere speed.  This is
    intended for calibrating the atmosphere scale, not for replacing the full
    orbit propagation.
    """
    radius_km = radius_body_km + altitude_km
    radius_m = radius_km * 1000.0
    mu_m3_s2 = mu_km3_s2 * 1.0e9

    circular_speed_m_s = np.sqrt(mu_m3_s2 / radius_m)
    atmosphere_speed_m_s = earth_rotation_rad_s * radius_m
    relative_speed_m_s = circular_speed_m_s - atmosphere_speed_m_s

    specific_drag_accel_m_s2 = (
        0.5
        * density_kg_m3
        * drag_coefficient
        * area_m2
        / spacecraft_mass_kg
        * relative_speed_m_s**2
    )

    # Circular-orbit energy balance: dE/dt = v * a_tangential,
    # E = -mu/(2a).  For a nearly circular orbit, altitude and semimajor-axis
    # decay rates are approximately equal.
    decay_rate_m_s = (
        -2.0
        * radius_m**2
        / mu_m3_s2
        * circular_speed_m_s
        * specific_drag_accel_m_s2
    )
    return decay_rate_m_s * seconds_per_day / 1000.0


def calibrate_environment_to_decay_rate(
    reference_altitude_km=initial_altitude_km,
    target_decay_rate_km_per_day=nominal_decay_rate_km_per_day,
    area_m2=area_ram_m2,
    spacecraft_mass_kg=mass_kg,
    drag_coefficient=cd,
    reference_altitude_model_km=400.0,
    scale_height_km=58.0,
    calibration_duration_days=1.0,
    target_final_altitude_km=None,
    apply=True,
):
    """Calibrate the density scale using the Orekit numerical propagator.

    The source challenge states that a nominal ram-face spacecraft at 330 km
    should decay at -0.512 km/day during solar maximum.  This function computes
    the density multiplier required for the existing exponential atmosphere to
    reproduce that reference using the same drag physics as the full simulation.

    Parameters
    ----------
    reference_altitude_km : float
        Altitude where the decay-rate reference is specified.
    target_decay_rate_km_per_day : float
        Desired altitude decay rate.  Negative values indicate decay.
    area_m2 : float
        Effective drag area for the reference configuration.
    spacecraft_mass_kg : float
        Spacecraft mass.
    drag_coefficient : float
        Drag coefficient.
    apply : bool
        If True, update the module-level density scale used by
        ``atmospheric_density_kg_m3``.

    Returns
    -------
    dict
        Calibration metadata including the scale factor, base decay rate, and
        calibrated decay rate.
    """
    global ATMOSPHERIC_DENSITY_SCALE

    from functions.orekit import propagate_segment

    duration_s = float(calibration_duration_days) * seconds_per_day
    if duration_s <= 0.0:
        raise ValueError("calibration_duration_days must be positive")
    if target_final_altitude_km is None:
        target_final_altitude_km = (
            reference_altitude_km
            + target_decay_rate_km_per_day * calibration_duration_days
        )

    radius_km = earth_radius_km + reference_altitude_km
    speed_km_s = np.sqrt(mu_earth_km3_s2 / radius_km)
    initial_state = np.array([radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0])

    def final_mean_sma_altitude(scale):
        _, states, _ = propagate_segment(
            initial_state,
            duration_s,
            min(3600.0, duration_s),
            area_m2=area_m2,
            spacecraft_mass_kg=spacecraft_mass_kg,
            drag_coefficient=drag_coefficient,
            rho0_kg_m3=3.5e-12 * scale,
            reference_altitude_km=reference_altitude_model_km,
            scale_height_km=scale_height_km,
        )
        radius = np.linalg.norm(states[:, :3], axis=1)
        speed_squared = np.sum(states[:, 3:] ** 2, axis=1)
        sma_altitude = -mu_earth_km3_s2 / (
            2.0 * (0.5 * speed_squared - mu_earth_km3_s2 / radius)
        ) - earth_radius_km
        tail_count = max(
            1,
            int(round(min(seconds_per_day, duration_s) / min(3600.0, duration_s))),
        )
        return float(np.mean(sma_altitude[-tail_count:]))

    lower_scale = 0.0
    upper_scale = 1.0
    while final_mean_sma_altitude(upper_scale) > target_final_altitude_km:
        upper_scale *= 2.0
        if upper_scale > 1.0e6:
            raise RuntimeError("Unable to bracket Orekit atmosphere calibration")

    for _ in range(24):
        density_scale = 0.5 * (lower_scale + upper_scale)
        if final_mean_sma_altitude(density_scale) > target_final_altitude_km:
            lower_scale = density_scale
        else:
            upper_scale = density_scale

    density_scale = 0.5 * (lower_scale + upper_scale)
    calibrated_final_altitude = final_mean_sma_altitude(density_scale)
    calibrated_decay_rate = (
        calibrated_final_altitude - reference_altitude_km
    ) / calibration_duration_days

    if apply:
        ATMOSPHERIC_DENSITY_SCALE = float(density_scale)

    base_density = base_atmospheric_density_kg_m3(reference_altitude_km)
    calibrated_density = density_scale * base_density
    base_final_altitude = final_mean_sma_altitude(1.0)
    base_decay_rate = (
        base_final_altitude - reference_altitude_km
    ) / calibration_duration_days

    return {
        "reference_altitude_km": float(reference_altitude_km),
        "target_decay_rate_km_per_day": float(target_decay_rate_km_per_day),
        "base_density_kg_m3": float(base_density),
        "calibrated_density_kg_m3": float(calibrated_density),
        "density_scale": float(density_scale),
        "base_decay_rate_km_per_day": float(base_decay_rate),
        "calibrated_decay_rate_km_per_day": float(calibrated_decay_rate),
        "target_final_altitude_km": float(target_final_altitude_km),
        "calibrated_final_altitude_km": float(calibrated_final_altitude),
        "calibration_duration_days": float(calibration_duration_days),
        "reference_altitude_model_km": float(reference_altitude_model_km),
        "scale_height_km": float(scale_height_km),
        "area_m2": float(area_m2),
        "mass_kg": float(spacecraft_mass_kg),
        "cd": float(drag_coefficient),
    }
