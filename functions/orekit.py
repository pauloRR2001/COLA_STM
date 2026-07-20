"""Python interface to the repository's Orekit numerical propagator."""

from __future__ import annotations

import csv
import io
import os
import subprocess
from pathlib import Path

import numpy as np

from constants import cd, isp_s, mass_kg

ROOT = Path(__file__).resolve().parents[1]
MAVEN = ROOT / "tools" / "apache-maven-3.9.12" / "bin" / "mvn.cmd"
MAIN_CLASS = "com.colastm.OrekitPropagationCli"
CLASSPATH_FILE = ROOT / "target" / "runtime-classpath.txt"


def _ensure_compiled() -> str:
    """Compile the Orekit bridge once and return the direct Java classpath."""
    source = ROOT / "src" / "main" / "java" / "com" / "colastm" / "OrekitPropagationCli.java"
    compiled = ROOT / "target" / "classes" / "com" / "colastm" / "OrekitPropagationCli.class"
    needs_compile = not compiled.exists() or compiled.stat().st_mtime < source.stat().st_mtime
    if needs_compile or not CLASSPATH_FILE.exists():
        command = [
            "cmd.exe", "/c", str(MAVEN), "-q", "compile",
            "dependency:build-classpath",
            f"-Dmdep.outputFile={CLASSPATH_FILE}",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError("Orekit bridge compilation failed.\n" + completed.stdout + "\n" + completed.stderr)
    dependencies = CLASSPATH_FILE.read_text(encoding="utf-8").strip()
    return os.pathsep.join((str(ROOT / "target" / "classes"), dependencies))


def _run_cli(arguments: list[str], expect_stm: bool) -> tuple[np.ndarray, np.ndarray | None]:
    command = ["java", "-cp", _ensure_compiled(), MAIN_CLASS, *arguments]
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError("Orekit propagation failed.\n" + completed.stdout + "\n" + completed.stderr)
    start = completed.stdout.find("time_s,")
    if start < 0:
        raise RuntimeError("Orekit output did not contain a CSV state history.")
    reader = csv.DictReader(io.StringIO(completed.stdout[start:]))
    rows = []
    stms = [] if expect_stm else None
    for row in reader:
        try:
            rows.append([float(row[key]) for key in (
                "time_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "mass_kg")])
            if expect_stm:
                stms.append([[float(row[f"phi_{i}_{j}"]) for j in range(6)] for i in range(6)])
        except (KeyError, TypeError, ValueError):
            break
    if not rows:
        raise RuntimeError("Orekit returned no numerical states.")
    return np.asarray(rows, dtype=float), (np.asarray(stms, dtype=float) if expect_stm else None)


def propagate_segment(initial_state_km: np.ndarray, duration_s: float, output_step_s: float, *,
                      area_m2: float, spacecraft_mass_kg: float = mass_kg,
                      drag_coefficient: float = cd, rho0_kg_m3: float = 3.5e-12,
                      reference_altitude_km: float = 400.0, scale_height_km: float = 58.0,
                      thrust_n: float = 0.0, specific_impulse_s: float = isp_s,
                      thrust_direction: str = "NONE", epoch_offset_s: float = 0.0,
                      return_stm: bool = False):
    state = np.asarray(initial_state_km, dtype=float)
    if state.shape != (6,):
        raise ValueError("initial_state_km must have shape (6,)")
    args = [*(f"{value * 1000.0:.17g}" for value in state[:3]),
            *(f"{value * 1000.0:.17g}" for value in state[3:]),
            f"{spacecraft_mass_kg:.17g}", f"{duration_s:.17g}", f"{output_step_s:.17g}",
            f"{area_m2:.17g}", f"{drag_coefficient:.17g}", f"{rho0_kg_m3:.17g}",
            f"{reference_altitude_km * 1000.0:.17g}", f"{scale_height_km * 1000.0:.17g}",
            f"{thrust_n:.17g}", f"{specific_impulse_s:.17g}", thrust_direction,
            f"{epoch_offset_s:.17g}", str(bool(return_stm)).lower()]
    raw, stms = _run_cli(args, return_stm)
    times = raw[:, 0] + epoch_offset_s
    states = np.column_stack((raw[:, 1:4] / 1000.0, raw[:, 4:7] / 1000.0))
    if return_stm:
        return times, states, raw[:, 7], stms
    return times, states, raw[:, 7]


def propagate_schedule(initial_state_km: np.ndarray, segments: list[dict], output_step_s: float, *,
                       initial_mass_kg: float = mass_kg, return_stm: bool = False):
    current_state = np.asarray(initial_state_km, dtype=float)
    current_mass = float(initial_mass_kg)
    epoch_offset = 0.0
    all_times, all_states, all_masses, all_stms = [], [], [], []
    cumulative_stm = np.eye(6)
    for segment in segments:
        duration = float(segment["duration_s"])
        propagated = propagate_segment(
            current_state, duration, min(output_step_s, abs(duration)),
            area_m2=float(segment["area_m2"]), spacecraft_mass_kg=current_mass,
            drag_coefficient=float(segment.get("cd", cd)),
            rho0_kg_m3=float(segment.get("rho0_kg_m3", 3.5e-12)),
            reference_altitude_km=float(segment.get("reference_altitude_km", 400.0)),
            scale_height_km=float(segment.get("scale_height_km", 58.0)),
            thrust_n=float(segment.get("thrust_n", 0.0)),
            specific_impulse_s=float(segment.get("isp_s", isp_s)),
            thrust_direction=str(segment.get("thrust_direction", "NONE")),
            epoch_offset_s=epoch_offset, return_stm=return_stm)
        if return_stm:
            times, states, masses, local_stms = propagated
            local_stms = np.einsum("nij,jk->nik", local_stms, cumulative_stm)
        else:
            times, states, masses = propagated
        if all_times:
            times, states, masses = times[1:], states[1:], masses[1:]
            if return_stm:
                local_stms = local_stms[1:]
        all_times.append(times); all_states.append(states); all_masses.append(masses)
        if return_stm:
            all_stms.append(local_stms)
            cumulative_stm = local_stms[-1]
        current_state = states[-1]; current_mass = masses[-1]; epoch_offset += duration
    result = (np.concatenate(all_times), np.vstack(all_states), np.concatenate(all_masses))
    if return_stm:
        return (*result, np.vstack(all_stms))
    return result
