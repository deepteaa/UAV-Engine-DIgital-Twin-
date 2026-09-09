"""
CSV Data Logger  (Version 6)
-----------------------------------------------------------------------
Roadmap reference: Layer 2 "Data Collection".

Every previous version printed sensor data to the terminal and then lost
it forever. That's fine for watching a demo, but useless for building a
dataset. This module appends one row per simulated second to a CSV file,
including the digital-twin comparison, the anomaly detector's verdict,
and the *ground-truth* fault label from the FaultScheduler - so later we
can measure how good the detector actually is against reality.
"""

import csv
import os
from datetime import datetime


class DataLogger:
    FIELDNAMES = [
        "timestamp",
        "sim_time_s",
        "mission_phase",
        "rpm",
        "load",
        "cooling_efficiency",
        "cht",
        "expected_cht",
        "residual",
        "engine_health_index",
        "health_status",
        "oil_temperature",
        "oil_pressure",
        "fuel_flow",
        "vibration",
        "fault_type",
    ]

    def __init__(self, filepath, overwrite=True):
        self.filepath = filepath

        mode = "w" if overwrite or not os.path.exists(filepath) else "a"
        write_header = mode == "w"

        self._file = open(filepath, mode, newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if write_header:
            self._writer.writeheader()

    def log(self, **kwargs):
        row = {field: kwargs.get(field, "") for field in self.FIELDNAMES}
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()
