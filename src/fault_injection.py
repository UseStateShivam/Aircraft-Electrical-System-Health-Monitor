"""
Controlled Fault Injection Engine for Aircraft Electrical Bus Time-Series Data.

Modifies normal synthetic operating data by injecting controlled electrical anomalies
and sensor malfunctions while maintaining strict, uncorrupted ground-truth labels.

Implements all 10 required fault types:
  1. Overvoltage
  2. Undervoltage
  3. Overcurrent
  4. Frequency Deviation (AC only)
  5. Voltage Spike (transient surge)
  6. Voltage Sag (transient dip)
  7. Current Spike (transient pulse)
  8. Sensor Dropout (missing values / NaN)
  9. Sensor Drift (gradual degradation)
  10. Combined / Multivariate Anomaly

DISCLAIMER: Educational simulation prototype. Controlled mathematical modifications.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import AppConfig


@dataclass
class FaultEvent:
    """Metadata describing a single injected fault event."""

    fault_id: str
    fault_type: str
    start_time: float
    end_time: float
    start_idx: int
    end_idx: int
    affected_variable: str
    severity: str
    magnitude_factor: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FaultInjector:
    """Engine for scheduling and injecting faults into electrical-bus time series."""

    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def inject_faults(
        self,
        df: pd.DataFrame,
        num_faults: Optional[int] = None,
        seed: Optional[int] = None,
        selected_faults: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[FaultEvent]]:
        """Inject a randomized sequence of controlled electrical and sensor faults.

        Parameters
        ----------
        df : pd.DataFrame
            Normal baseline electrical time-series dataframe.
        num_faults : int, optional
            Number of fault events to inject. Defaults to config setting.
        seed : int, optional
            Random seed for placement and magnitude decisions.
        selected_faults : list of str, optional
            Specific subset of fault types to inject.

        Returns
        -------
        Tuple[pd.DataFrame, List[FaultEvent]]
            Modified DataFrame containing injected anomalies and updated ground-truth columns,
            plus the list of FaultEvent metadata records.
        """
        data = df.copy()
        active_seed = seed if seed is not None else self.config.seed
        rng = np.random.RandomState(active_seed)

        bus_type = str(data["bus_type"].iloc[0]).lower()
        cfg_fault = self.config.fault_injection
        total_faults = num_faults if num_faults is not None else int(cfg_fault.get("num_faults", 10))

        # Filter supported fault types (exclude frequency deviation for DC)
        available_types = selected_faults or cfg_fault.get("supported_faults", [])
        if bus_type == "dc":
            available_types = [f for f in available_types if f != "frequency_deviation"]

        n_samples = len(data)
        sampling_rate = self.config.sampling_rate_hz
        total_duration = data["timestamp"].iloc[-1]

        # Allocate non-overlapping time slots across the timeline
        # Leave buffer at start and end for steady baseline (adapt for short datasets)
        start_buffer = min(int(15.0 * sampling_rate), int(0.10 * n_samples))
        end_buffer = min(int(10.0 * sampling_rate), int(0.08 * n_samples))
        available_span = max(10, n_samples - start_buffer - end_buffer)

        if available_span < total_faults * 10:
            total_faults = max(2, available_span // 15)

        slot_size = max(6, available_span // total_faults)
        events: List[FaultEvent] = []

        severities = ["mild", "moderate", "severe"]
        sev_factors = cfg_fault.get(
            "severity_factors", {"mild": 1.10, "moderate": 1.25, "severe": 1.45}
        )

        for i in range(total_faults):
            fault_id = f"FLT_{i + 1:02d}"
            fault_type = available_types[i % len(available_types)]
            severity = rng.choice(severities)
            factor = float(sev_factors.get(severity, 1.25))

            # Determine slot window
            slot_start = start_buffer + i * slot_size
            slot_end = min(n_samples - end_buffer, slot_start + slot_size)
            curr_slot_len = slot_end - slot_start

            # Determine fault duration based on fault nature
            if fault_type in ("voltage_spike", "voltage_sag", "current_spike"):
                dur_samples = rng.randint(2, max(3, int(0.8 * sampling_rate)))
            elif fault_type == "sensor_drift":
                dur_samples = rng.randint(max(4, int(15 * sampling_rate)), max(5, int(35 * sampling_rate)))
            elif fault_type == "sensor_dropout":
                dur_samples = rng.randint(max(3, int(3 * sampling_rate)), max(4, int(10 * sampling_rate)))
            else:
                dur_samples = rng.randint(max(3, int(6 * sampling_rate)), max(4, int(18 * sampling_rate)))

            # Bound duration within current slot
            dur_samples = min(dur_samples, max(2, curr_slot_len - 2))
            min_start = slot_start + 1
            max_start = slot_end - dur_samples
            if max_start <= min_start:
                event_start_idx = slot_start
            else:
                event_start_idx = rng.randint(min_start, max_start)

            event_end_idx = min(n_samples, event_start_idx + dur_samples)

            # Inject the specific fault into data
            event = self._apply_fault(
                df=data,
                fault_id=fault_id,
                fault_type=fault_type,
                start_idx=event_start_idx,
                end_idx=event_end_idx,
                severity=severity,
                factor=factor,
                bus_type=bus_type,
                rng=rng,
            )
            events.append(event)

        # Recalculate apparent power with any updated voltage/current
        data["apparent_power"] = np.round(data["voltage"] * data["current"], 2)

        self.logger.info(
            f"Successfully injected {len(events)} fault events into {bus_type.upper()} dataset."
        )
        return data, events

    def _apply_fault(
        self,
        df: pd.DataFrame,
        fault_id: str,
        fault_type: str,
        start_idx: int,
        end_idx: int,
        severity: str,
        factor: float,
        bus_type: str,
        rng: np.random.RandomState,
    ) -> FaultEvent:
        """Apply mathematical anomaly to the specified slice and label ground truth."""
        sl = df.index[start_idx:end_idx]
        t_start = float(df["timestamp"].iloc[start_idx])
        t_end = float(df["timestamp"].iloc[end_idx - 1])

        affected_var = "voltage"
        description = f"{severity.capitalize()} {fault_type}"

        # 1. Overvoltage
        if fault_type == "overvoltage":
            affected_var = "voltage"
            df.loc[sl, "voltage"] = df.loc[sl, "voltage"] * factor

        # 2. Undervoltage
        elif fault_type == "undervoltage":
            affected_var = "voltage"
            undervolt_factor = 1.0 / factor
            df.loc[sl, "voltage"] = df.loc[sl, "voltage"] * undervolt_factor

        # 3. Overcurrent
        elif fault_type == "overcurrent":
            affected_var = "current"
            # Increase current significantly above operating ceiling
            df.loc[sl, "current"] = df.loc[sl, "current"] * (factor * 1.15)

        # 4. Frequency Deviation (AC only)
        elif fault_type == "frequency_deviation":
            affected_var = "frequency"
            if bus_type == "ac":
                # Shift frequency up or down by 8 - 25 Hz
                direction = rng.choice([-1.0, 1.0])
                deviation_hz = (factor - 1.0) * 45.0 + 8.0
                df.loc[sl, "frequency"] = df.loc[sl, "frequency"] + (direction * deviation_hz)

        # 5. Voltage Spike (sudden transient surge)
        elif fault_type == "voltage_spike":
            affected_var = "voltage"
            spike_factor = factor * 1.20
            df.loc[sl, "voltage"] = df.loc[sl, "voltage"] * spike_factor

        # 6. Voltage Sag (sudden transient drop)
        elif fault_type == "voltage_sag":
            affected_var = "voltage"
            sag_factor = max(0.40, 1.0 - (factor - 1.0) * 1.5)
            df.loc[sl, "voltage"] = df.loc[sl, "voltage"] * sag_factor

        # 7. Current Spike
        elif fault_type == "current_spike":
            affected_var = "current"
            c_spike_factor = factor * 1.35
            df.loc[sl, "current"] = df.loc[sl, "current"] * c_spike_factor

        # 8. Sensor Dropout (simulate sensor loss as NaN)
        elif fault_type == "sensor_dropout":
            affected_var = rng.choice(["voltage", "current"])
            df.loc[sl, affected_var] = np.nan
            description = f"Sensor dropout on {affected_var} (missing data/NaN)"

        # 9. Sensor Drift (gradual linear divergence)
        elif fault_type == "sensor_drift":
            affected_var = "voltage"
            ramp = np.linspace(0.0, (factor - 1.0) * 15.0, end_idx - start_idx)
            direction = rng.choice([-1.0, 1.0])
            df.loc[sl, "voltage"] = df.loc[sl, "voltage"] + (direction * ramp)

        # 10. Combined / Multivariate Anomaly
        elif fault_type == "combined_fault":
            affected_var = "voltage,current,frequency" if bus_type == "ac" else "voltage,current"
            # Voltage sags while current surges (e.g., partial short circuit / overload)
            df.loc[sl, "voltage"] = df.loc[sl, "voltage"] * 0.82
            df.loc[sl, "current"] = df.loc[sl, "current"] * 1.40
            if bus_type == "ac":
                df.loc[sl, "frequency"] = df.loc[sl, "frequency"] - 5.5

        else:
            raise ValueError(f"Unknown fault_type: {fault_type}")

        # Update ground truth columns
        df.loc[sl, "true_fault"] = 1
        df.loc[sl, "fault_type"] = fault_type
        df.loc[sl, "fault_id"] = fault_id
        df.loc[sl, "severity"] = severity

        return FaultEvent(
            fault_id=fault_id,
            fault_type=fault_type,
            start_time=t_start,
            end_time=t_end,
            start_idx=start_idx,
            end_idx=end_idx,
            affected_variable=affected_var,
            severity=severity,
            magnitude_factor=factor,
            description=description,
        )
