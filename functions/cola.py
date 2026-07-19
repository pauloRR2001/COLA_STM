from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


AccelerationModel = Callable[[np.ndarray, float], np.ndarray]


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


def two_body_acceleration(mu_km3_s2: float) -> AccelerationModel:
    def model(state: np.ndarray, _time: float) -> np.ndarray:
        r = state[:3]
        return -mu_km3_s2 * r / np.linalg.norm(r) ** 3

    return model


def combined_acceleration(*models: AccelerationModel) -> AccelerationModel:
    def model(state: np.ndarray, time: float) -> np.ndarray:
        total = np.zeros(3)
        for item in models:
            total += item(state, time)
        return total

    return model


def gravity_jacobian(state: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    r = state[:3]
    radius = np.linalg.norm(r)
    identity = np.eye(3)
    gradient = mu_km3_s2 * (3.0 * np.outer(r, r) / radius**5 - identity / radius**3)
    matrix = np.zeros((6, 6))
    matrix[:3, 3:] = identity
    matrix[3:, :3] = gradient
    return matrix


def _derivative(
    augmented: np.ndarray,
    time: float,
    acceleration_model: AccelerationModel,
    mu_km3_s2: float,
) -> np.ndarray:
    state = augmented[:6]
    phi = augmented[6:].reshape(6, 6)
    state_rate = np.hstack((state[3:], acceleration_model(state, time)))
    phi_rate = gravity_jacobian(state, mu_km3_s2) @ phi
    return np.hstack((state_rate, phi_rate.ravel()))


def _rk4_step(
    augmented: np.ndarray,
    time: float,
    step: float,
    acceleration_model: AccelerationModel,
    mu_km3_s2: float,
) -> np.ndarray:
    f = lambda x, t: _derivative(x, t, acceleration_model, mu_km3_s2)
    k1 = f(augmented, time)
    k2 = f(augmented + 0.5 * step * k1, time + 0.5 * step)
    k3 = f(augmented + 0.5 * step * k2, time + 0.5 * step)
    k4 = f(augmented + step * k3, time + step)
    return augmented + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def propagate(
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    duration_seconds: float,
    step_seconds: float,
    acceleration_model: AccelerationModel,
    mu_km3_s2: float,
) -> PropagationResult:
    count = int(abs(duration_seconds) / abs(step_seconds)) + 1
    times = np.linspace(0.0, duration_seconds, count)
    augmented = np.hstack((initial_state, np.eye(6).ravel()))
    states = np.zeros((count, 6))
    covariances = np.zeros((count, 6, 6))
    states[0] = initial_state
    covariances[0] = initial_covariance

    for index in range(1, count):
        dt = times[index] - times[index - 1]
        augmented = _rk4_step(
            augmented,
            times[index - 1],
            dt,
            acceleration_model,
            mu_km3_s2,
        )
        phi = augmented[6:].reshape(6, 6)
        states[index] = augmented[:6]
        covariances[index] = phi @ initial_covariance @ phi.T

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
