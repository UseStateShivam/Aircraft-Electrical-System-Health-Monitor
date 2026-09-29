"""
Engineering Threshold Monitoring and Fault Classification Engine.

Implements conventional aviation electrical limit monitoring using strict configurable
threshold envelopes. Assigns standardized alert severity levels:
  - NORMAL: Operating within normal tolerances
  - WARNING: Approaching operational boundary
  - CRITICAL: Exceeding safety threshold limits
  - SENSOR_FAULT: Missing or corrupted sensor telemetry

DISCLAIMER: Educational prototype rules; not certified avionics thresholds.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import ACThresholds, AppConfig, DCThresholds


class EngineeringThresholdMonitor:
    """Conventional rule-based limit checking engine for aircraft electrical buses.

    Parameters
    ----------
    config : AppConfig
        Application configuration containing thresholds for AC and DC buses.
    logger : logging.Logger, optional
        Logger instance.
    """

    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def monitor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate engineering threshold rules across the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed electrical sensor dataframe.

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with threshold monitoring outputs:
            - 'rule_anomaly': int (0 = normal, 1 = threshold violation)
            - 'rule_alert_level': str ('NORMAL', 'WARNING', 'CRITICAL', 'SENSOR_FAULT')
            - 'rule_fault_type': str ('NORMAL', 'OVERVOLTAGE', 'UNDERVOLTAGE', etc.)
        """
        data = df.copy()
        bus_type = str(data["bus_type"].iloc[0]).lower()

        n_samples = len(data)
        rule_anomaly = np.zeros(n_samples, dtype=int)
        rule_alert_level = np.array(["NORMAL"] * n_samples, dtype=object)
        rule_fault_type = np.array(["NORMAL"] * n_samples, dtype=object)

        # 1. Sensor Fault Evaluation (highest priority rule)
        # Check raw sensor dropout or impossible values
        is_sensor_fault = (
            data.get("sensor_dropout_flag", data["voltage"].isna() | data["current"].isna()) == 1
        )
        if "impossible_value_flag" in data.columns:
            is_sensor_fault = is_sensor_fault | (data["impossible_value_flag"] == 1)

        rule_anomaly[is_sensor_fault] = 1
        rule_alert_level[is_sensor_fault] = "SENSOR_FAULT"
        rule_fault_type[is_sensor_fault] = "SENSOR_DROPOUT"

        # Apply electrical thresholds only on non-sensor-fault samples
        valid_mask = ~is_sensor_fault
        v = data["voltage"].fillna(0.0).values
        c = data["current"].fillna(0.0).values

        if bus_type == "ac":
            thresh: ACThresholds = self.config.thresholds_ac
            f = data["frequency"].fillna(0.0).values

            # Critical overvoltage
            crit_ov = valid_mask & (v >= thresh.voltage_critical_high)
            rule_anomaly[crit_ov] = 1
            rule_alert_level[crit_ov] = "CRITICAL"
            rule_fault_type[crit_ov] = "OVERVOLTAGE"

            # Critical undervoltage
            crit_uv = valid_mask & (v <= thresh.voltage_critical_low)
            rule_anomaly[crit_uv] = 1
            rule_alert_level[crit_uv] = "CRITICAL"
            rule_fault_type[crit_uv] = "UNDERVOLTAGE"

            # Critical overcurrent
            crit_oc = valid_mask & (c >= thresh.current_critical_high)
            rule_anomaly[crit_oc] = 1
            rule_alert_level[crit_oc] = "CRITICAL"
            rule_fault_type[crit_oc] = "OVERCURRENT"

            # Critical frequency deviations
            crit_f_high = valid_mask & (f >= thresh.frequency_critical_high)
            rule_anomaly[crit_f_high] = 1
            rule_alert_level[crit_f_high] = "CRITICAL"
            rule_fault_type[crit_f_high] = "FREQUENCY_DEVIATION"

            crit_f_low = valid_mask & (f <= thresh.frequency_critical_low)
            rule_anomaly[crit_f_low] = 1
            rule_alert_level[crit_f_low] = "CRITICAL"
            rule_fault_type[crit_f_low] = "FREQUENCY_DEVIATION"

            # Warning levels (only if not already marked CRITICAL or SENSOR_FAULT)
            norm_mask = valid_mask & (rule_alert_level == "NORMAL")

            warn_ov = norm_mask & (v >= thresh.voltage_warning_high) & (v < thresh.voltage_critical_high)
            rule_anomaly[warn_ov] = 1
            rule_alert_level[warn_ov] = "WARNING"
            rule_fault_type[warn_ov] = "OVERVOLTAGE"

            warn_uv = norm_mask & (v <= thresh.voltage_warning_low) & (v > thresh.voltage_critical_low)
            rule_anomaly[warn_uv] = 1
            rule_alert_level[warn_uv] = "WARNING"
            rule_fault_type[warn_uv] = "UNDERVOLTAGE"

            warn_oc = norm_mask & (c >= thresh.current_warning_high) & (c < thresh.current_critical_high)
            rule_anomaly[warn_oc] = 1
            rule_alert_level[warn_oc] = "WARNING"
            rule_fault_type[warn_oc] = "OVERCURRENT"

            warn_f_high = norm_mask & (f >= thresh.frequency_warning_high) & (f < thresh.frequency_critical_high)
            rule_anomaly[warn_f_high] = 1
            rule_alert_level[warn_f_high] = "WARNING"
            rule_fault_type[warn_f_high] = "FREQUENCY_DEVIATION"

            warn_f_low = norm_mask & (f <= thresh.frequency_warning_low) & (f > thresh.frequency_critical_low)
            rule_anomaly[warn_f_low] = 1
            rule_alert_level[warn_f_low] = "WARNING"
            rule_fault_type[warn_f_low] = "FREQUENCY_DEVIATION"

        elif bus_type == "dc":
            thresh_dc: DCThresholds = self.config.thresholds_dc

            # Critical overvoltage
            crit_ov = valid_mask & (v >= thresh_dc.voltage_critical_high)
            rule_anomaly[crit_ov] = 1
            rule_alert_level[crit_ov] = "CRITICAL"
            rule_fault_type[crit_ov] = "OVERVOLTAGE"

            # Critical undervoltage
            crit_uv = valid_mask & (v <= thresh_dc.voltage_critical_low)
            rule_anomaly[crit_uv] = 1
            rule_alert_level[crit_uv] = "CRITICAL"
            rule_fault_type[crit_uv] = "UNDERVOLTAGE"

            # Critical overcurrent
            crit_oc = valid_mask & (c >= thresh_dc.current_critical_high)
            rule_anomaly[crit_oc] = 1
            rule_alert_level[crit_oc] = "CRITICAL"
            rule_fault_type[crit_oc] = "OVERCURRENT"

            # Warnings
            norm_mask = valid_mask & (rule_alert_level == "NORMAL")

            warn_ov = norm_mask & (v >= thresh_dc.voltage_warning_high) & (v < thresh_dc.voltage_critical_high)
            rule_anomaly[warn_ov] = 1
            rule_alert_level[warn_ov] = "WARNING"
            rule_fault_type[warn_ov] = "OVERVOLTAGE"

            warn_uv = norm_mask & (v <= thresh_dc.voltage_warning_low) & (v > thresh_dc.voltage_critical_low)
            rule_anomaly[warn_uv] = 1
            rule_alert_level[warn_uv] = "WARNING"
            rule_fault_type[warn_uv] = "UNDERVOLTAGE"

            warn_oc = norm_mask & (c >= thresh_dc.current_warning_high) & (c < thresh_dc.current_critical_high)
            rule_anomaly[warn_oc] = 1
            rule_alert_level[warn_oc] = "WARNING"
            rule_fault_type[warn_oc] = "OVERCURRENT"

        data["rule_anomaly"] = rule_anomaly
        data["rule_alert_level"] = rule_alert_level
        data["rule_fault_type"] = rule_fault_type

        violations = int(np.sum(rule_anomaly))
        self.logger.info(
            f"Threshold monitor completed for {bus_type.upper()}: "
            f"{violations}/{n_samples} samples violated thresholds."
        )
        return data
