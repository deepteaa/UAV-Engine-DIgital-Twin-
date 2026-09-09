"""
Fault Scheduler  (Version 7)
-----------------------------------------------------------------------
Roadmap reference: Layer 1 "Modification Required" + Layer 4 "Faults we
need to simulate" + Modification 2 ("Don't hardcode faults to mission
phases").

The old simulator did this:

    if phase == "CLIMB":
        cooling_efficiency = 0.6

...which means every CLIMB is automatically faulty. That's unrealistic
and useless for training data, because the label is 100% correlated with
the phase instead of with an actual developing fault.

This module lets faults:
  - start at a specific simulation second, independent of mission phase
  - ramp gradually from a healthy value to a faulty value over time
  - run several fault types at once, each on its own timeline
  - carry a ground-truth label for CSV/ML use, separate from whatever
    the anomaly detector *thinks* is happening
"""


class FaultDefinition:
    def __init__(self, name, parameter, healthy_value, faulty_value,
                 start_time, ramp_duration, active_threshold=0.05):
        """
        name:             ground-truth label, e.g. "COOLING_DEGRADATION"
        parameter:        which simulated parameter this affects, e.g.
                           "cooling_efficiency", "oil_pressure_factor",
                           "vibration_offset"
        healthy_value:    value of the parameter with no fault
        faulty_value:     value once the fault is fully developed
        start_time:       simulation second at which degradation begins
        ramp_duration:    seconds it takes to go from healthy -> faulty
        active_threshold: how far into the ramp (0-1) before we call the
                           fault "active" for labelling purposes
        """
        self.name = name
        self.parameter = parameter
        self.healthy_value = healthy_value
        self.faulty_value = faulty_value
        self.start_time = start_time
        self.ramp_duration = max(ramp_duration, 1e-6)
        self.active_threshold = active_threshold

    def progress(self, sim_time):
        """0.0 = not started yet, 1.0 = fully developed."""
        if sim_time <= self.start_time:
            return 0.0
        elapsed = sim_time - self.start_time
        return min(1.0, elapsed / self.ramp_duration)

    def value_at(self, sim_time):
        p = self.progress(sim_time)
        return self.healthy_value + (self.faulty_value - self.healthy_value) * p

    def is_active(self, sim_time):
        return self.progress(sim_time) >= self.active_threshold


class FaultScheduler:
    def __init__(self):
        self.definitions = []

    def add_fault(self, fault_definition):
        self.definitions.append(fault_definition)

    def get_value(self, parameter, sim_time, healthy_default):
        """
        Current value of `parameter` at `sim_time`. If more than one
        fault targets the same parameter, the one furthest from the
        healthy default wins (keeps the physics model simple for now).
        """
        candidates = [
            d.value_at(sim_time)
            for d in self.definitions
            if d.parameter == parameter
        ]
        if not candidates:
            return healthy_default
        return max(candidates, key=lambda v: abs(v - healthy_default))

    def active_fault_labels(self, sim_time):
        """List of ground-truth fault names active at this instant."""
        active = [d.name for d in self.definitions if d.is_active(sim_time)]
        return active if active else ["NORMAL"]

    def active_fault_label_string(self, sim_time):
        return "+".join(self.active_fault_labels(sim_time))
