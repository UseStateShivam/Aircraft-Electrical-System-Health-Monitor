"""
Unit tests for Engineering Threshold Monitor.

Validates conventional rule checking, severity alert hierarchy, and AC vs DC isolation.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.threshold_monitor import EngineeringThresholdMonitor


@pytest.fixture
def config():
    return load_config("config/config.yaml")


def test_detects_obvious_ac_voltage_faults(config):
    """Test that threshold monitor detects critical overvoltage and undervoltage."""
    monitor = EngineeringThresholdMonitor(config=config)

    # Synthetic 3-sample test
    df = pd.DataFrame(
        {
            "bus_type": ["ac", "ac", "ac"],
            "voltage": [115.0, 135.0, 95.0],  # normal, critical high, critical low
            "current": [15.0, 15.0, 15.0],
            "frequency": [400.0, 400.0, 400.0],
            "sensor_dropout_flag": [0, 0, 0],
        }
    )

    res = monitor.monitor(df)
    assert res.loc[0, "rule_anomaly"] == 0
    assert res.loc[0, "rule_alert_level"] == "NORMAL"

    assert res.loc[1, "rule_anomaly"] == 1
    assert res.loc[1, "rule_alert_level"] == "CRITICAL"
    assert res.loc[1, "rule_fault_type"] == "OVERVOLTAGE"

    assert res.loc[2, "rule_anomaly"] == 1
    assert res.loc[2, "rule_alert_level"] == "CRITICAL"
    assert res.loc[2, "rule_fault_type"] == "UNDERVOLTAGE"


def test_dc_mode_ignores_frequency_rules(config):
    """Test that DC mode with NaN frequency does not raise false frequency alerts."""
    monitor = EngineeringThresholdMonitor(config=config)

    df = pd.DataFrame(
        {
            "bus_type": ["dc", "dc"],
            "voltage": [28.0, 28.0],
            "current": [35.0, 35.0],
            "frequency": [np.nan, np.nan],  # DC mode has NaN frequency
            "sensor_dropout_flag": [0, 0],
        }
    )

    res = monitor.monitor(df)
    # Neither should be flagged as anomaly or frequency error
    assert (res["rule_anomaly"] == 0).all()
    assert (res["rule_alert_level"] == "NORMAL").all()


def test_sensor_dropout_flags_sensor_fault(config):
    """Test that missing sensor data produces SENSOR_FAULT alert."""
    monitor = EngineeringThresholdMonitor(config=config)

    df = pd.DataFrame(
        {
            "bus_type": ["ac"],
            "voltage": [np.nan],
            "current": [15.0],
            "frequency": [400.0],
            "sensor_dropout_flag": [1],
        }
    )

    res = monitor.monitor(df)
    assert res.loc[0, "rule_anomaly"] == 1
    assert res.loc[0, "rule_alert_level"] == "SENSOR_FAULT"
    assert res.loc[0, "rule_fault_type"] == "SENSOR_DROPOUT"
