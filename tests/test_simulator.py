"""
Unit tests for Synthetic Electrical Bus Simulator.

Validates data shape, column presence, physical plausibility, and AC/DC mode handling.
"""

import numpy as np
import pytest

from src.config import load_config
from src.simulator import ElectricalBusSimulator


@pytest.fixture
def config():
    return load_config("config/config.yaml")


def test_simulator_sample_count_and_columns(config):
    """Test that simulator generates expected sample count and required columns."""
    sim = ElectricalBusSimulator(config=config)
    fs = 10.0
    dur = 5.0
    df = sim.generate(duration_seconds=dur, sampling_rate_hz=fs, bus_type="ac", seed=10)

    expected_samples = int(dur * fs)
    assert len(df) == expected_samples

    required_cols = [
        "timestamp",
        "bus_type",
        "operating_condition",
        "load_level",
        "voltage",
        "current",
        "frequency",
        "apparent_power",
        "true_fault",
        "fault_type",
        "fault_id",
        "severity",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_ac_bus_electrical_properties(config):
    """Test AC bus nominal envelope and non-NaN frequency."""
    sim = ElectricalBusSimulator(config=config)
    df = sim.generate(duration_seconds=10.0, bus_type="ac", seed=42)

    assert df["bus_type"].iloc[0] == "ac"
    # Voltage should oscillate around 115V RMS (within 105 - 125V)
    assert df["voltage"].mean() == pytest.approx(115.0, abs=4.0)
    # Frequency should oscillate around 400Hz (within 398 - 402Hz)
    assert df["frequency"].notna().all()
    assert df["frequency"].mean() == pytest.approx(400.0, abs=1.5)
    # Ground truth for raw baseline should be all 0 (normal)
    assert (df["true_fault"] == 0).all()


def test_dc_bus_electrical_properties(config):
    """Test DC bus nominal envelope and verify frequency is explicitly NaN."""
    sim = ElectricalBusSimulator(config=config)
    df = sim.generate(duration_seconds=10.0, bus_type="dc", seed=42)

    assert df["bus_type"].iloc[0] == "dc"
    # DC Voltage should oscillate around 28V (within 26 - 30V)
    assert df["voltage"].mean() == pytest.approx(28.0, abs=1.5)
    # Frequency MUST be NaN for DC (DC does not have a 400Hz AC frequency)
    assert df["frequency"].isna().all(), "DC bus should not have AC frequency values"
    # Current should be positive
    assert (df["current"] > 0).all()
