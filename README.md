# COLA_STM

A Python-based astrodynamics toolkit for constellation deployment and collision avoidance maneuver planning.

The project combines analytical astrodynamics, custom mission scheduling algorithms, and orbit propagation to study low Earth orbit (LEO) constellation operations. The software is structured around reusable simulation components and is designed to support mission analysis, trade studies, and collision avoidance strategy development.

Although several simplified analytical models are included for rapid design-space exploration, the project is intended to interface with Orekit for high-fidelity orbit propagation and force modeling.

---

## Features

- Analytical and numerical orbital mechanics models
- J2-induced RAAN drift analysis
- Constellation deployment and orbital phasing
- In-plane slotting using differential mean motion
- Continuous low-thrust collision avoidance
- Differential drag collision avoidance
- State Transition Matrix (STM) propagation
- Covariance propagation
- Synthetic Conjunction Data Message (CDM) generation
- Collision probability estimation
- Lead-time optimization for avoidance maneuvers
- Modular simulation architecture

---

## Repository Structure

```text
COLA_STM/
│
├── challenge_1_part_a.py      # RAAN drift analysis and deployment trade studies
├── challenge_1_part_b.py      # In-plane constellation slotting
│
├── challenge_2_part_1.py      # Continuous low-thrust collision avoidance
├── challenge_2_part_2.py      # Differential drag collision avoidance
│
├── constants.py               # Mission and spacecraft constants
│
├── functions/
│   ├── __init__.py
│   ├── cola.py                # Shared propagation and conjunction utilities
│   └── tle.py                 # TLE utilities
│
└── orekit-data/               # Orekit data package
```

---

## Simulation Overview

### RAAN Drift Analysis

Evaluates constellation deployment strategies using natural J2 nodal precession.

Outputs include:

- Altitude evolution
- RAAN rate
- Cumulative RAAN drift
- Strategy comparison
- Deployment ΔV trade studies

---

### Constellation Slotting

Demonstrates in-plane deployment by temporarily modifying orbital mean motion.

Outputs include:

- Relative mean anomaly evolution
- Temporary phasing orbits
- Slot acquisition
- Constellation geometry
- Phasing ΔV

---

### Continuous Low-Thrust Collision Avoidance

Simulates collision avoidance using continuous low-thrust maneuvers.

Features include:

- State propagation
- STM propagation
- Covariance propagation
- Lead-time search
- Burn scheduling
- Recovery maneuver
- Synthetic CDM generation

---

### Differential Drag Collision Avoidance

Simulates collision avoidance by modifying spacecraft ballistic coefficient.

Features include:

- Atmospheric drag model
- Variable drag area scheduling
- Lead-time optimization
- Along-track timing adjustment
- Synthetic CDM generation

---

## Physics Models

Current models include:

- Two-body dynamics
- J2 secular perturbations
- Continuous low-thrust acceleration
- Exponential atmospheric density
- Differential drag
- Linear covariance propagation
- State Transition Matrix propagation

The repository is designed so higher-fidelity Orekit propagators can replace analytical models with minimal changes to the overall architecture.

---

## Requirements

- Python 3.12+
- NumPy
- Matplotlib
- Requests
- Orekit (optional for high-fidelity propagation)

Install dependencies with:

```bash
pip install numpy matplotlib requests
```

---

## Running the Simulations

### RAAN Drift

```bash
python challenge_1_part_a.py
```

### Constellation Slotting

```bash
python challenge_1_part_b.py
```

### Low-Thrust Collision Avoidance

```bash
python challenge_2_part_1.py
```

### Differential Drag Collision Avoidance

```bash
python challenge_2_part_2.py
```

---

## Outputs

Depending on the simulation, outputs include:

- Mission summary tables
- Orbit and trajectory visualizations
- RAAN evolution plots
- Constellation phasing plots
- Relative distance histories
- Collision probability histories
- Maneuver optimization results
- Synthetic Conjunction Data Messages (CDMs)

---

## Future Improvements

- Full Orekit numerical propagation
- High-order gravity models
- Solar radiation pressure
- Third-body perturbations
- Harris–Priester and NRLMSISE-00 atmosphere models
- Foster/Alfano collision probability formulations
- Monte Carlo covariance analysis
- Multi-spacecraft scheduling optimization
- Automated maneuver planning
- CCSDS-compliant CDM generation

---

## License

This repository is provided for research, educational, and engineering development purposes.