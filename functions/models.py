
"""Common data models used throughout the COLA_STM project."""
from functions.state_machine import SpacecraftState
def create_spacecraft(identifier="SC"):
    return {
        "id": identifier,
        "orbit": {"r": None, "v": None},
        "attitude": {"q": None, "omega": None},
        "vehicle": {
            "mass": None,
            "cd": None,
            "drag_area": None,
        },
        "operations": {
            "state": SpacecraftState.NOMINAL,
            "recovering": False,
            "recovered": False,
            "mission_lost": False,
        },
    }
def create_collision():
    return {
        "tca": None,
        "distance": None,
        "probability": None,
        "warning": False,
        "collision": False,
        "maneuver_required": False,
    }
def create_maneuver():
    return {
        "epoch": None,
        "delta_v": None,
        "magnitude": None,
        "executed": False,
    }
def create_conjunction():
    return {
        "primary": create_spacecraft("Primary"),
        "secondary": create_spacecraft("Secondary"),
        "collision": create_collision(),
        "maneuver": create_maneuver(),
    }
