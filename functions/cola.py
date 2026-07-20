from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from constants import area_ram_m2, cd, drag_reference_altitude_km, drag_reference_density_kg_m3, drag_scale_height_km
from functions.orekit import propagate_schedule


@dataclass
class PropagationResult:
    times: np.ndarray
    states: np.ndarray
    covariances: np.ndarray


@dataclass
class ConjunctionResult:
    index: int
    tca_seconds: float
    miss_distance_km: float
    relative_position_eci_km: np.ndarray
    relative_velocity_eci_km_s: np.ndarray
    relative_position_rtn_km: np.ndarray
    relative_velocity_rtn_km_s: np.ndarray
    combined_position_covariance_rtn_km2: np.ndarray
    collision_probability: float
    distances_km: np.ndarray


def rtn_basis(state: np.ndarray) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    r_hat = r / np.linalg.norm(r)
    h_hat = np.cross(r, v)
    h_hat /= np.linalg.norm(h_hat)
    t_hat = np.cross(h_hat, r_hat)
    return np.column_stack((r_hat, t_hat, h_hat))


def two_body_acceleration(mu_km3_s2: float) -> dict:
    """Return the default Orekit force-model configuration.

    The name is retained for compatibility with the challenge scripts. Orekit
    always supplies the central field through its 20x20 spherical-harmonic
    gravity model, so this function no longer creates a Python acceleration.
    """
    return {
        "engine": "orekit",
        "mu_km3_s2": mu_km3_s2,
        "area_m2": area_ram_m2,
        "cd": cd,
        "rho0_kg_m3": drag_reference_density_kg_m3,
        "reference_altitude_km": drag_reference_altitude_km,
        "scale_height_km": drag_scale_height_km,
        "thrust_windows": [],
        "area_windows": [],
    }


def combined_acceleration(*models: dict) -> dict:
    combined = two_body_acceleration(0.0)
    for model in models:
        if not isinstance(model, dict):
            raise TypeError("Custom Python acceleration callbacks are no longer supported; use Orekit configuration dictionaries.")
        for key, value in model.items():
            if key in ("thrust_windows", "area_windows"):
                combined[key].extend(value)
            else:
                combined[key] = value
    return combined


def propagate(
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    duration_seconds: float,
    step_seconds: float,
    acceleration_model: dict,
    mu_km3_s2: float,
) -> PropagationResult:
    if not isinstance(acceleration_model, dict) or acceleration_model.get("engine") != "orekit":
        raise TypeError("Propagation requires an Orekit force-model configuration.")

    direction = 1.0 if duration_seconds >= 0.0 else -1.0
    end_time = duration_seconds
    boundaries = {0.0, end_time}
    for key in ("thrust_windows", "area_windows"):
        for start, stop, *_ in acceleration_model.get(key, []):
            if min(0.0, end_time) < start < max(0.0, end_time):
                boundaries.add(float(start))
            if min(0.0, end_time) < stop < max(0.0, end_time):
                boundaries.add(float(stop))
    ordered = sorted(boundaries, reverse=direction < 0.0)
    segments = []
    for start, stop in zip(ordered[:-1], ordered[1:]):
        midpoint = 0.5 * (start + stop)
        area = float(acceleration_model.get("area_m2", area_ram_m2))
        for window_start, window_stop, scheduled_area in acceleration_model.get("area_windows", []):
            if min(window_start, window_stop) <= midpoint < max(window_start, window_stop):
                area = float(scheduled_area)
                break
        thrust = 0.0
        thrust_direction = "NONE"
        for window_start, window_stop, scheduled_thrust, scheduled_direction in acceleration_model.get("thrust_windows", []):
            if min(window_start, window_stop) <= midpoint < max(window_start, window_stop):
                thrust = float(scheduled_thrust)
                thrust_direction = str(scheduled_direction)
                break
        segments.append({
            "duration_s": stop - start,
            "area_m2": area,
            "cd": acceleration_model.get("cd", cd),
            "rho0_kg_m3": acceleration_model.get("rho0_kg_m3", drag_reference_density_kg_m3),
            "reference_altitude_km": acceleration_model.get("reference_altitude_km", drag_reference_altitude_km),
            "scale_height_km": acceleration_model.get("scale_height_km", drag_scale_height_km),
            "thrust_n": thrust,
            "thrust_direction": thrust_direction,
        })

    times, states, _masses, stms = propagate_schedule(
        initial_state, segments, abs(step_seconds), return_stm=True
    )
    initial_covariance = np.asarray(initial_covariance, dtype=float)
    covariances = np.einsum("nij,jk,nlk->nil", stms, initial_covariance, stms)
    return PropagationResult(times, states, covariances)


def estimate_collision_probability(
    miss_rtn_km: np.ndarray,
    covariance_rtn_km2: np.ndarray,
    hard_body_radius_km: float,
) -> float:
    miss_2d = miss_rtn_km[:2]
    covariance_2d = covariance_rtn_km2[:2, :2]
    determinant = np.linalg.det(covariance_2d)
    if determinant <= 0.0:
        return 0.0
    inverse = np.linalg.inv(covariance_2d)
    exponent = -0.5 * miss_2d @ inverse @ miss_2d
    probability = hard_body_radius_km**2 / (2.0 * np.sqrt(determinant))
    probability *= np.exp(exponent)
    return float(np.clip(probability, 0.0, 1.0))


def assess_conjunction(
    primary: PropagationResult,
    secondary: PropagationResult,
    hard_body_radius_km: float,
) -> ConjunctionResult:
    relative_positions = secondary.states[:, :3] - primary.states[:, :3]
    relative_velocities = secondary.states[:, 3:] - primary.states[:, 3:]
    distances = np.linalg.norm(relative_positions, axis=1)
    index = int(np.argmin(distances))
    basis = rtn_basis(primary.states[index])
    relative_rtn = basis.T @ relative_positions[index]
    relative_velocity_rtn = basis.T @ relative_velocities[index]
    combined_covariance = primary.covariances[index] + secondary.covariances[index]
    covariance_rtn = basis.T @ combined_covariance[:3, :3] @ basis
    probability = estimate_collision_probability(
        relative_rtn,
        covariance_rtn,
        hard_body_radius_km,
    )
    return ConjunctionResult(
        index=index,
        tca_seconds=float(primary.times[index]),
        miss_distance_km=float(distances[index]),
        relative_position_eci_km=relative_positions[index],
        relative_velocity_eci_km_s=relative_velocities[index],
        relative_position_rtn_km=relative_rtn,
        relative_velocity_rtn_km_s=relative_velocity_rtn,
        combined_position_covariance_rtn_km2=covariance_rtn,
        collision_probability=probability,
        distances_km=distances,
    )


def assess_conjunction_at_time(
    primary: PropagationResult,
    secondary: PropagationResult,
    tca_seconds: float,
    hard_body_radius_km: float,
) -> ConjunctionResult:
    """Assess a conjunction at a known CDM TCA instead of searching globally."""
    index = int(np.argmin(np.abs(primary.times - tca_seconds)))
    relative_position = secondary.states[index, :3] - primary.states[index, :3]
    relative_velocity = secondary.states[index, 3:] - primary.states[index, 3:]
    basis = rtn_basis(primary.states[index])
    relative_rtn = basis.T @ relative_position
    relative_velocity_rtn = basis.T @ relative_velocity
    combined_covariance = primary.covariances[index] + secondary.covariances[index]
    covariance_rtn = basis.T @ combined_covariance[:3, :3] @ basis
    probability = estimate_collision_probability(
        relative_rtn,
        covariance_rtn,
        hard_body_radius_km,
    )
    distances = np.linalg.norm(secondary.states[:, :3] - primary.states[:, :3], axis=1)
    return ConjunctionResult(
        index=index,
        tca_seconds=float(primary.times[index]),
        miss_distance_km=float(np.linalg.norm(relative_position)),
        relative_position_eci_km=relative_position,
        relative_velocity_eci_km_s=relative_velocity,
        relative_position_rtn_km=relative_rtn,
        relative_velocity_rtn_km_s=relative_velocity_rtn,
        combined_position_covariance_rtn_km2=covariance_rtn,
        collision_probability=probability,
        distances_km=distances,
    )


def create_synthetic_encounter(
    mu_km3_s2: float,
    earth_radius_km: float,
    altitude_km: float,
    inclination_deg: float,
    miss_rtn_km: Iterable[float],
    relative_velocity_rtn_km_s: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    radius = earth_radius_km + altitude_km
    speed = np.sqrt(mu_km3_s2 / radius)
    inclination = np.radians(inclination_deg)
    primary = np.array(
        [radius, 0.0, 0.0, 0.0, speed * np.cos(inclination), speed * np.sin(inclination)]
    )
    basis = rtn_basis(primary)
    secondary = primary + np.hstack(
        (basis @ np.asarray(miss_rtn_km), basis @ np.asarray(relative_velocity_rtn_km_s))
    )
    return primary, secondary


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
