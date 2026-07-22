"""Compute Challenge 3 instantaneous decay and point-of-no-return timing.

The atmospheric density at 330 km is inferred from the challenge-provided
nominal ram-face decay rate.  That inferred density anchors the same
58-km-scale-height exponential atmosphere already used by the repository.
"""

from __future__ import annotations

import math

# Challenge baseline
MASS_KG = 420.0
CD = 2.2
AREA_RAM_M2 = 0.65
AREA_TUMBLE_M2 = 9.5
THRUST_N = 35.0e-3
REFERENCE_ALTITUDE_KM = 330.0
NOMINAL_RAM_DECAY_KM_DAY = -0.512
SCALE_HEIGHT_KM = 58.0

# Earth constants
MU_M3_S2 = 3.986004418e14
EARTH_RADIUS_M = 6_378_137.0
EARTH_ROTATION_RAD_S = 7.2921150e-5
SECONDS_PER_DAY = 86_400.0


def circular_speeds(altitude_km: float) -> tuple[float, float]:
    """Return inertial circular speed and equatorial atmosphere-relative speed."""
    radius_m = EARTH_RADIUS_M + altitude_km * 1000.0
    circular_speed = math.sqrt(MU_M3_S2 / radius_m)
    atmosphere_speed = EARTH_ROTATION_RAD_S * radius_m
    return circular_speed, circular_speed - atmosphere_speed


def density_from_reference_decay() -> float:
    """Infer rho(330 km) from the specified ram-face SMA decay rate."""
    radius_m = EARTH_RADIUS_M + REFERENCE_ALTITUDE_KM * 1000.0
    circular_speed, relative_speed = circular_speeds(REFERENCE_ALTITUDE_KM)
    da_dt_m_s = NOMINAL_RAM_DECAY_KM_DAY * 1000.0 / SECONDS_PER_DAY

    # da/dt = -(r^2/mu) * v_c * rho * Cd*A/m * v_rel^2
    return (
        -da_dt_m_s
        * MU_M3_S2
        * MASS_KG
        / (radius_m**2 * circular_speed * CD * AREA_RAM_M2 * relative_speed**2)
    )


def density_kg_m3(altitude_km: float, rho_ref: float) -> float:
    return rho_ref * math.exp((REFERENCE_ALTITUDE_KM - altitude_km) / SCALE_HEIGHT_KM)


def drag_sma_rate_km_day(altitude_km: float, area_m2: float, rho_ref: float) -> float:
    radius_m = EARTH_RADIUS_M + altitude_km * 1000.0
    circular_speed, relative_speed = circular_speeds(altitude_km)
    rho = density_kg_m3(altitude_km, rho_ref)
    drag_accel = 0.5 * rho * CD * area_m2 / MASS_KG * relative_speed**2
    da_dt_m_s = -2.0 * radius_m**2 / MU_M3_S2 * circular_speed * drag_accel
    return da_dt_m_s * SECONDS_PER_DAY / 1000.0


def thrust_sma_rate_km_day(altitude_km: float) -> float:
    """Maximum circular-orbit SMA gain from continuous tangential HET thrust."""
    radius_m = EARTH_RADIUS_M + altitude_km * 1000.0
    thrust_accel = THRUST_N / MASS_KG
    da_dt_m_s = 2.0 * radius_m ** 1.5 / math.sqrt(MU_M3_S2) * thrust_accel
    return da_dt_m_s * SECONDS_PER_DAY / 1000.0


def find_no_return_altitude_km(rho_ref: float) -> float:
    """Solve |da_drag/dt| = da_thrust/dt for the tumbling configuration."""
    low_km = 80.0
    high_km = REFERENCE_ALTITUDE_KM

    def balance(h_km: float) -> float:
        return thrust_sma_rate_km_day(h_km) + drag_sma_rate_km_day(
            h_km, AREA_TUMBLE_M2, rho_ref
        )

    if balance(low_km) > 0.0 or balance(high_km) < 0.0:
        raise RuntimeError("Point of no return is not bracketed.")

    for _ in range(100):
        mid_km = 0.5 * (low_km + high_km)
        if balance(mid_km) > 0.0:
            high_km = mid_km
        else:
            low_km = mid_km
    return 0.5 * (low_km + high_km)


def exponential_decay_time_days(target_altitude_km: float, rho_ref: float) -> float:
    """Integrate drag-only tumbling descent using RK4 in altitude."""
    altitude_km = REFERENCE_ALTITUDE_KM
    time_days = 0.0
    dt_days = 1.0e-4  # 8.64 s

    def dhdt(h_km: float) -> float:
        return drag_sma_rate_km_day(h_km, AREA_TUMBLE_M2, rho_ref)

    while altitude_km > target_altitude_km:
        remaining = altitude_km - target_altitude_km
        local_rate = abs(dhdt(altitude_km))
        step = min(dt_days, remaining / local_rate)

        k1 = dhdt(altitude_km)
        k2 = dhdt(altitude_km + 0.5 * step * k1)
        k3 = dhdt(altitude_km + 0.5 * step * k2)
        k4 = dhdt(altitude_km + step * k3)
        altitude_km += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        time_days += step

    return time_days


def main() -> None:
    rho_330 = density_from_reference_decay()
    tumble_decay_330 = drag_sma_rate_km_day(
        REFERENCE_ALTITUDE_KM, AREA_TUMBLE_M2, rho_330
    )
    thrust_gain_330 = thrust_sma_rate_km_day(REFERENCE_ALTITUDE_KM)
    no_return_altitude = find_no_return_altitude_km(rho_330)
    no_return_drag = drag_sma_rate_km_day(
        no_return_altitude, AREA_TUMBLE_M2, rho_330
    )
    no_return_thrust = thrust_sma_rate_km_day(no_return_altitude)

    fixed_rate_time_days = (
        REFERENCE_ALTITUDE_KM - no_return_altitude
    ) / abs(tumble_decay_330)
    exponential_time_days = exponential_decay_time_days(no_return_altitude, rho_330)

    print("Challenge 3 point-of-no-return analysis")
    print("---------------------------------------")
    print(f"Inferred density at 330 km:        {rho_330:.6e} kg/m^3")
    print(f"Ram decay check at 330 km:         {drag_sma_rate_km_day(330.0, AREA_RAM_M2, rho_330):.6f} km/day")
    print(f"Tumbling SMA decay at 330 km:      {tumble_decay_330:.6f} km/day")
    print(f"Maximum HET SMA gain at 330 km:    {thrust_gain_330:.6f} km/day")
    print(f"Point-of-no-return altitude:       {no_return_altitude:.6f} km")
    print(f"Drag rate at no return:            {no_return_drag:.6f} km/day")
    print(f"Thrust rate at no return:          {no_return_thrust:.6f} km/day")
    print(f"Time using fixed 330-km decay:     {fixed_rate_time_days:.6f} days")
    print(f"Time using exponential atmosphere: {exponential_time_days:.6f} days")
    print(f"Fixed-rate time in hours:          {fixed_rate_time_days * 24.0:.3f} h")
    print(f"Exponential-model time in hours:   {exponential_time_days * 24.0:.3f} h")


if __name__ == "__main__":
    main()
