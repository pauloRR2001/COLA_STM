
import numpy as np
from constants import area_ram_m2,area_tumble_m2
from functions.state_machine import SpacecraftState
EARTH_ROTATION_RAD_S=7.2921159e-5
def atmospheric_density_kg_m3(altitude_km):
    rho0=3.5e-12
    h0=400.0
    H=58.0
    return rho0*np.exp(np.clip(-(altitude_km-h0)/H,-50,50))
def drag_area_from_state(state):
    if state in (SpacecraftState.TUMBLING,SpacecraftState.RECOVERY):
        return area_tumble_m2
    return area_ram_m2
