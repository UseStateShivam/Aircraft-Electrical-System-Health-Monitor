"""
Combined Aircraft Electrical Health Assessment and Condition Monitoring Engine.

Fuses engineering threshold limit checks, Isolation Forest machine-learning anomaly
outputs, and sensor telemetry quality flags into a transparent, multi-level health status:
  - HEALTHY: Normal operation within tolerances
  - DEGRADED: Early/subtle multivariate anomaly flagged by ML before threshold breach
  - WARNING: Approaching operational safety thresholds
  - CRITICAL: Exceeding severe limits or persistent compound failure
  - SENSOR_FAULT: Hardware telemetry loss or invalid sensor readings

Computes a deterministic, point-deduction Health Score (0-100) and compiles structured alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import AppConfig


@dataclass
class AlertRecord:
    """Individual condition monitoring alert item."""

    timestamp: float
    bus_type: str
    parameter: str
    observed_value: float
    expected_value: float
    threshold: str
    fault_type: str
    severity: str
    ml_anomaly_flag: int
    health_status: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 2),
            "bus_type": self.bus_type.upper(),
            "parameter": self.parameter,
            "observed_value": round(self.observed_value, 2) if not np.isnan(self.observed_value) else None,
            "expected_value": round(self.expected_value, 2),
            "threshold": self.threshold,
            "fault_type": self.fault_type,
            "severity": self.severity,
            "ml_anomaly_flag": self.ml_anomaly_flag,
            "health_status": self.health_status,
            "description": self.description,
        }


class IntegratedHealthMonitor:
    """Health monitor integrating rule checks, ML predictions, and sensor quality.

    Parameters
    ----------
    config : AppConfig
        Configuration containing scoring rules and bus parameters.
    logger : logging.Logger, optional
        Logger instance.
    """

    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def evaluate_health(
        self,
        df_processed: pd.DataFrame,
        df_rules: pd.DataFrame,
        df_ml: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[AlertRecord]]:
        """Combine all monitoring tiers to evaluate system health across time.

        Parameters
        ----------
        df_processed : pd.DataFrame
            Preprocessed sensor dataframe with engineered features and quality flags.
        df_rules : pd.DataFrame
            Outputs from EngineeringThresholdMonitor.
        df_ml : pd.DataFrame
            Outputs from IsolationForestDetector.

        Returns
        -------
        Tuple[pd.DataFrame, List[AlertRecord]]
            1. Enriched master DataFrame with unified health columns:
               - 'final_anomaly' (int: 0 or 1)
               - 'health_status' (str)
               - 'diagnosed_fault_type' (str)
               - 'health_score' (float: 0.0 to 100.0)
            2. List of AlertRecord instances for condition-based maintenance logs.
        """
        combined = df_processed.copy()

        # Merge rule columns
        combined["rule_anomaly"] = df_rules["rule_anomaly"]
        combined["rule_alert_level"] = df_rules["rule_alert_level"]
        combined["rule_fault_type"] = df_rules["rule_fault_type"]

        # Merge ML columns
        combined["ml_anomaly"] = df_ml["ml_anomaly"]
        combined["ml_anomaly_score"] = df_ml["ml_anomaly_score"]
        combined["ml_raw_score"] = df_ml["ml_raw_score"]

        n_samples = len(combined)
        bus_type = str(combined["bus_type"].iloc[0]).lower()

        # Unified anomaly flag: flagged if rule violation OR ML anomaly OR sensor fault
        final_anomaly = (
            (combined["rule_anomaly"] == 1)
            | (combined["ml_anomaly"] == 1)
            | (combined.get("sensor_dropout_flag", 0) == 1)
        ).astype(int)

        # Health status assignment hierarchy
        health_status = np.array(["HEALTHY"] * n_samples, dtype=object)
        diagnosed_fault = np.array(["NORMAL"] * n_samples, dtype=object)

        # 1. ML-only anomalies -> DEGRADED
        ml_only = (combined["ml_anomaly"] == 1) & (combined["rule_anomaly"] == 0)
        health_status[ml_only] = "DEGRADED"
        diagnosed_fault[ml_only] = "UNSUPERVISED_ANOMALY"

        # 2. Rule Warnings -> WARNING
        rule_warn = combined["rule_alert_level"] == "WARNING"
        health_status[rule_warn] = "WARNING"
        diagnosed_fault[rule_warn] = combined.loc[rule_warn, "rule_fault_type"]

        # 3. Rule Criticals -> CRITICAL
        rule_crit = combined["rule_alert_level"] == "CRITICAL"
        health_status[rule_crit] = "CRITICAL"
        diagnosed_fault[rule_crit] = combined.loc[rule_crit, "rule_fault_type"]

        # 4. Sensor faults -> SENSOR_FAULT
        sensor_fault = (combined.get("sensor_dropout_flag", 0) == 1) | (
            combined.get("impossible_value_flag", 0) == 1
        )
        health_status[sensor_fault] = "SENSOR_FAULT"
        diagnosed_fault[sensor_fault] = "SENSOR_DROPOUT"

        # 5. Continuous Health Score Calculation (0 - 100)
        penalties = self.config.health_score.get("penalties", {})
        p_sensor = float(penalties.get("sensor_fault", 40.0))
        p_crit = float(penalties.get("threshold_critical", 35.0))
        p_warn = float(penalties.get("threshold_warning", 12.0))
        p_ml = float(penalties.get("ml_anomaly", 15.0))

        raw_health_score = np.full(n_samples, 100.0)

        # Apply transparent deductions
        for i in range(n_samples):
            deduction = 0.0
            if sensor_fault.iloc[i]:
                deduction += p_sensor
            if rule_crit.iloc[i]:
                deduction += p_crit
            elif rule_warn.iloc[i]:
                deduction += p_warn

            if combined["ml_anomaly"].iloc[i] == 1:
                # Scale ML deduction by anomaly score
                deduction += p_ml * combined["ml_anomaly_score"].iloc[i]

            raw_health_score[i] = max(0.0, 100.0 - deduction)

        # Smooth health score to avoid 1-sample jitter
        # Exponential moving average filter with alpha=0.3
        smoothed_score = np.copy(raw_health_score)
        alpha = 0.35
        for i in range(1, n_samples):
            smoothed_score[i] = alpha * raw_health_score[i] + (1 - alpha) * smoothed_score[i - 1]

        combined["final_anomaly"] = final_anomaly
        combined["health_status"] = health_status
        combined["diagnosed_fault_type"] = diagnosed_fault
        combined["health_score"] = np.round(smoothed_score, 1)

        # Compile structured maintenance alert records
        alerts = self._compile_alerts(combined, bus_type)

        self.logger.info(
            f"Integrated health assessment completed. "
            f"Anomalous samples: {np.sum(final_anomaly)}/{n_samples}. "
            f"Alerts logged: {len(alerts)}."
        )
        return combined, alerts

    def _compile_alerts(self, df: pd.DataFrame, bus_type: str) -> List[AlertRecord]:
        """Extract continuous alert episodes for maintenance reporting."""
        alerts: List[AlertRecord] = []
        nominal_v = (
            self.config.ac_bus.nominal_voltage
            if bus_type == "ac"
            else self.config.dc_bus.nominal_voltage
        )

        anom_rows = df[df["final_anomaly"] == 1]
        for idx, row in anom_rows.iterrows():
            param = "voltage"
            obs_val = row["voltage"]
            exp_val = nominal_v
            thresh_str = "Envelope"

            if row["rule_fault_type"] == "OVERCURRENT":
                param = "current"
                obs_val = row["current"]
                exp_val = (
                    self.config.ac_bus.base_current
                    if bus_type == "ac"
                    else self.config.dc_bus.base_current
                )
                thresh_str = "Overcurrent Limit"
            elif row["rule_fault_type"] == "FREQUENCY_DEVIATION":
                param = "frequency"
                obs_val = row["frequency"]
                exp_val = self.config.ac_bus.nominal_frequency
                thresh_str = "Frequency Envelope"

            alert = AlertRecord(
                timestamp=float(row["timestamp"]),
                bus_type=bus_type,
                parameter=param,
                observed_value=float(obs_val) if pd.notna(obs_val) else np.nan,
                expected_value=float(exp_val),
                threshold=thresh_str,
                fault_type=str(row["diagnosed_fault_type"]),
                severity=str(row["health_status"]),
                ml_anomaly_flag=int(row["ml_anomaly"]),
                health_status=str(row["health_status"]),
                description=f"Status: {row['health_status']}, Diagnosed: {row['diagnosed_fault_type']}",
            )
            alerts.append(alert)

        return alerts
