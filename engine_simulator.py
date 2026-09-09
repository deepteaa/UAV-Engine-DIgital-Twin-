import random
import time

from digital_twin import DigitalTwin
from fault_scheduler import FaultScheduler, FaultDefinition
from data_logger import DataLogger
from engine_physics import EngineState, tick, detect_anomaly, engine_health_index

print("UAV ENGINE MISSION SIMULATOR STARTED  (v7 - fault scheduler + CSV logging)")
print("---------------------------------------------------------------------------")

# --------------------------------
# RUN SETTINGS
# --------------------------------

USE_FIXED_SEED = True    # Modification 3: reproducible runs for debugging
REALTIME = True          # False = skip the 1s sleep -> generate datasets fast
LOG_FILE = "engine_mission_log.csv"

if USE_FIXED_SEED:
    random.seed(42)

# --------------------------------
# MISSION PHASES
# --------------------------------

mission_phases = [
    ("IDLE", 2000, 0.2),
    ("TAKEOFF", 5500, 1.0),
    ("CLIMB", 4800, 0.8),
    ("CRUISE", 4000, 0.6),
    ("DESCENT", 2800, 0.3),
    ("IDLE", 2000, 0.2)
]

SECONDS_PER_PHASE = 10

# --------------------------------
# FAULT SCHEDULE  (Version 7)
# --------------------------------
# Faults are now scheduled against simulation time, not mission phase.
# CLIMB is no longer *automatically* faulty - whether a fault is active
# depends only on the clock below, so the label isn't just a copy of the
# phase name. Feel free to comment faults out to generate clean NORMAL
# missions, or move start_time / ramp_duration to reshape the dataset.

scheduler = FaultScheduler()

scheduler.add_fault(FaultDefinition(
    name="COOLING_DEGRADATION",
    parameter="cooling_efficiency",
    healthy_value=1.0,
    faulty_value=0.6,
    start_time=15,       # begins mid-way through TAKEOFF
    ramp_duration=20,    # takes 20s to fully develop (not instant on/off)
))

scheduler.add_fault(FaultDefinition(
    name="OIL_PRESSURE_DEGRADATION",
    parameter="oil_pressure_factor",
    healthy_value=1.0,
    faulty_value=0.75,
    start_time=40,
    ramp_duration=15,
))

scheduler.add_fault(FaultDefinition(
    name="VIBRATION_FAULT",
    parameter="vibration_offset",
    healthy_value=0.0,
    faulty_value=0.35,
    start_time=48,
    ramp_duration=10,
))

# --------------------------------
# INITIAL ENGINE STATE
# --------------------------------

engine_state = EngineState(rpm=2000, cht=120, oil_temp=70)

twin = DigitalTwin()
sim_time = 0  # cumulative seconds since mission start; drives the fault schedule

logger = DataLogger(LOG_FILE)

# --------------------------------
# START MISSION
# --------------------------------

for phase, target_rpm, load in mission_phases:

    print("\n")
    print("================================")
    print(f"MISSION PHASE: {phase}")
    print("================================")

    for second in range(SECONDS_PER_PHASE):

        sim_time += 1

        # --------------------------------
        # FAULT-AFFECTED PARAMETERS  (Version 7)
        # --------------------------------

        cooling_efficiency = scheduler.get_value("cooling_efficiency", sim_time, 1.0)
        oil_pressure_factor = scheduler.get_value("oil_pressure_factor", sim_time, 1.0)
        vibration_offset = scheduler.get_value("vibration_offset", sim_time, 0.0)

        # --------------------------------
        # RUN ONE TICK OF ENGINE PHYSICS  (now in engine_physics.py)
        # --------------------------------

        readings = tick(
            engine_state, target_rpm, load,
            cooling_efficiency, oil_pressure_factor, vibration_offset,
        )

        # Update healthy digital twin & compare
        expected_cht = twin.update(readings["rpm"], load)
        residual = readings["cht"] - expected_cht

        # --------------------------------
        # ANOMALY DETECTION + ENGINE HEALTH INDEX  (now in engine_physics.py)
        # --------------------------------

        status = detect_anomaly(engine_state, residual)
        ehi = engine_health_index(residual)

        # --------------------------------
        # GROUND-TRUTH FAULT LABEL (for CSV / future ML training)
        # --------------------------------

        fault_type = scheduler.active_fault_label_string(sim_time)

        # --------------------------------
        # DISPLAY DATA
        # --------------------------------

        print("\n--- ENGINE SENSOR DATA ---")
        print(f"Sim Time: {sim_time}s | Phase: {phase} ({second + 1}/{SECONDS_PER_PHASE}s)")
        print(f"RPM: {readings['rpm']:.0f}")
        print(f"Cooling Efficiency: {cooling_efficiency * 100:.0f}%")
        print(f"CHT: {readings['cht']:.2f} °C | Twin Expected: {expected_cht:.2f} °C | Residual: {residual:.2f} °C")
        print(f"Engine Health Index: {ehi:.1f}/100 | Status: {status}")
        print(f"Oil Temperature: {readings['oil_temp']:.2f} °C | Oil Pressure: {readings['oil_pressure']:.2f} bar")
        print(f"Fuel Flow: {readings['fuel_flow']:.2f} | Vibration: {readings['vibration']:.3f}")
        print(f"Ground-truth fault label: {fault_type}")

        # --------------------------------
        # LOG TO CSV  (Version 6)
        # --------------------------------

        logger.log(
            sim_time_s=sim_time,
            mission_phase=phase,
            rpm=round(readings["rpm"], 1),
            load=load,
            cooling_efficiency=round(cooling_efficiency, 3),
            cht=round(readings["cht"], 2),
            expected_cht=round(expected_cht, 2),
            residual=round(residual, 2),
            engine_health_index=round(ehi, 1),
            health_status=status,
            oil_temperature=round(readings["oil_temp"], 2),
            oil_pressure=round(readings["oil_pressure"], 2),
            fuel_flow=round(readings["fuel_flow"], 2),
            vibration=round(readings["vibration"], 3),
            fault_type=fault_type,
        )

        if REALTIME:
            time.sleep(1)

logger.close()
print("\nMISSION COMPLETED")
print(f"Full labelled dataset written to: {LOG_FILE}")
