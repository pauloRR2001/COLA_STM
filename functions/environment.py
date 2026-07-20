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
    apply=True,
):
    """Calibrate the density scale to match a reference altitude-decay rate.

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

    base_density = base_atmospheric_density_kg_m3(reference_altitude_km)
    base_decay_rate = circular_drag_decay_rate_km_per_day(
        reference_altitude_km,
        base_density,
        area_m2=area_m2,
        spacecraft_mass_kg=spacecraft_mass_kg,
        drag_coefficient=drag_coefficient,
    )

    if np.isclose(base_decay_rate, 0.0):
        raise ValueError("Base decay rate is zero; cannot calibrate atmosphere scale.")

    density_scale = target_decay_rate_km_per_day / base_decay_rate

    if density_scale <= 0.0:
        raise ValueError(
            "Atmosphere calibration requires target and base decay rates to have "
            "the same sign."
        )

    if apply:
        ATMOSPHERIC_DENSITY_SCALE = float(density_scale)

    calibrated_density = density_scale * base_density
    calibrated_decay_rate = circular_drag_decay_rate_km_per_day(
        reference_altitude_km,
        calibrated_density,
        area_m2=area_m2,
        spacecraft_mass_kg=spacecraft_mass_kg,
        drag_coefficient=drag_coefficient,
    )

    return {
        "reference_altitude_km": float(reference_altitude_km),
        "target_decay_rate_km_per_day": float(target_decay_rate_km_per_day),
        "base_density_kg_m3": float(base_density),
        "calibrated_density_kg_m3": float(calibrated_density),
        "density_scale": float(density_scale),
        "base_decay_rate_km_per_day": float(base_decay_rate),
        "calibrated_decay_rate_km_per_day": float(calibrated_decay_rate),
        "area_m2": float(area_m2),
        "mass_kg": float(spacecraft_mass_kg),
        "cd": float(drag_coefficient),
    }
