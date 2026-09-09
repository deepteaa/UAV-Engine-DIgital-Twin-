"""
Engine Physics  (shared by engine_simulator.py and the dashboard)
-----------------------------------------------------------------------
Roadmap reference: Modification 1 ("Separate components" - engine
physics, mission, faults, digital twin, and detection should eventually
live in separate files instead of one big script).

This module owns the actual math: given the current engine state and a
mission phase's target RPM/load, compute the next second's sensor
readings, run the same anomaly-detection thresholds, and compute the
Engine Health Index. Both the terminal simulator and the Streamlit
dashboard import this module, so they can never quietly drift into two
different "engines" that disagree with each other.
"""

import random


class EngineState:
    """Mutable engine state carried between ticks."""

    def __init__(self, rpm=2000, cht=120, oil_temp=70):
        self.rpm = rpm
        self.cht = cht
        self.oil_temp = oil_temp
        self.anomaly_count = 0


def tick(state, target_rpm, load, cooling_efficiency, oil_pressure_factor, vibration_offset):
    """
    Advance the engine by one simulated second and return a dict of the
    resulting sensor readings. Mutates `state` in place. Does NOT touch
    the digital twin, fault scheduler, or CSV logging - callers own those.

    NOTE: the order of random.* calls below matches the original
    engine_simulator.py exactly, so a fixed random.seed() still produces
    identical numbers as before this refactor.
    """

    # RPM dynamics
    if state.rpm < target_rpm:
        state.rpm += 200
    elif state.rpm > target_rpm:
        state.rpm -= 200
    state.rpm += random.randint(-30, 30)

    # Heat generation / cooling / CHT
    heat_generation = (state.rpm * 0.02) + (load * 30)
    cooling_effect = 40 * cooling_efficiency
    target_cht = 80 + heat_generation - cooling_effect
    state.cht = state.cht + (target_cht - state.cht) * 0.1
    state.cht += random.uniform(-1, 1)

    # Oil temperature
    target_oil_temp = 50 + state.rpm * 0.008 + load * 10
    state.oil_temp = state.oil_temp + (target_oil_temp - state.oil_temp) * 0.08
    state.oil_temp += random.uniform(-0.5, 0.5)

    # Oil pressure (fault-affected)
    oil_pressure = (2 + state.rpm * 0.0005) * oil_pressure_factor
    oil_pressure += random.uniform(-0.15, 0.15)

    # Fuel flow
    fuel_flow = 3 + state.rpm * 0.002 + load * 5
    fuel_flow += random.uniform(-0.3, 0.3)

    # Vibration (fault-affected)
    vibration = 0.2 + load * 0.1 + vibration_offset
    vibration += random.uniform(-0.03, 0.03)

    return {
        "rpm": state.rpm,
        "cht": state.cht,
        "oil_temp": state.oil_temp,
        "oil_pressure": oil_pressure,
        "fuel_flow": fuel_flow,
        "vibration": vibration,
    }


def detect_anomaly(state, residual):
    """Same threshold / consecutive-count logic used everywhere."""
    if abs(residual) < 5:
        status = "NORMAL"
        state.anomaly_count = 0
    elif abs(residual) < 10:
        status = "CAUTION"
        state.anomaly_count = 0
    else:
        state.anomaly_count += 1
        status = "ANOMALY DETECTED" if state.anomaly_count >= 3 else "SUSPECTED ANOMALY"
    return status


def engine_health_index(residual):
    thermal_penalty = max(0, (abs(residual) - 3) * 5)
    return max(0, 100 - thermal_penalty)
