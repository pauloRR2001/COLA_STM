"""Simple spacecraft operational state machine for Challenge 3."""

from enum import Enum


class SpacecraftState(Enum):
    """Operational spacecraft states."""

    NOMINAL = "Nominal"
    TUMBLING = "Tumbling"
    RECOVERY = "Recovery"
    RECOVERED = "Recovered"
    MISSION_LOSS = "Mission Loss"


def update_state(data: dict) -> SpacecraftState:
    """Return the spacecraft operational state from status information.

    Supported dictionary keys are:
        omega_mag: angular-rate magnitude in deg/s
        recovering: True while recovery is underway
        recovered: True once recovery is complete
        mission_lost: True when the mission cannot be recovered
    """

    if data.get("mission_lost", False):
        return SpacecraftState.MISSION_LOSS

    if data.get("recovered", False):
        return SpacecraftState.RECOVERED

    if data.get("recovering", False):
        return SpacecraftState.RECOVERY

    if data.get("omega_mag", 0.0) > 1.0:
        return SpacecraftState.TUMBLING

    return SpacecraftState.NOMINAL
