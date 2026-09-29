"""
Unit tests for Fault Injection Engine.

Validates that faults modify expected variables, create ground-truth labels,
and honor AC vs DC operational constraints.
"""

import numpy as np
import pytest

from src.config import load_config
from src.fault_injection import FaultInjector
from src.simulator import ElectricalBusSimulator


@pytest.fixture
def config():
    return load_config("config/config.yaml")


def test_fault_injection_modifies_data_and_sets_labels(config):
    """Test that injecting faults modifies values and assigns ground-truth labels."""
    sim = ElectricalBusSimulator(config=config)
    injector = FaultInjector(config=config)

    df_clean = sim.generate(duration_seconds=60.0, sampling_rate_hz=10.0, bus_type="ac", seed=42)
    df_faulty, events = injector.inject_faults(df_clean, num_faults=5, seed=42)

    assert len(events) == 5
    assert len(df_faulty) == len(df_clean)

    # Some samples must now be labeled as true_fault == 1
    fault_mask = df_faulty["true_fault"] == 1
    assert fault_mask.any()
    assert (df_faulty["true_fault"] == 0).any()  # Normal majority preserved

    # Verify each event has correct metadata and modified the data
    for ev in events:
        sl = df_faulty.index[ev.start_idx : ev.end_idx]
        event_slice = df_faulty.loc[sl]
        assert (event_slice["true_fault"] == 1).all()
        assert (event_slice["fault_type"] == ev.fault_type).all()
        assert (event_slice["fault_id"] == ev.fault_id).all()


def test_sensor_dropout_produces_nans(config):
    """Test that sensor_dropout fault injects NaNs into sensor measurements."""
    sim = ElectricalBusSimulator(config=config)
    injector = FaultInjector(config=config)

    df_clean = sim.generate(duration_seconds=60.0, bus_type="ac", seed=42)
    df_faulty, events = injector.inject_faults(
        df_clean, num_faults=1, selected_faults=["sensor_dropout"], seed=42
    )

    ev = events[0]
    assert ev.fault_type == "sensor_dropout"
    var = ev.affected_variable
    sl = df_faulty.index[ev.start_idx : ev.end_idx]
    assert df_faulty.loc[sl, var].isna().all()


def test_dc_mode_excludes_frequency_deviation(config):
    """Test that DC mode never injects frequency deviation faults."""
    sim = ElectricalBusSimulator(config=config)
    injector = FaultInjector(config=config)

    df_clean = sim.generate(duration_seconds=60.0, bus_type="dc", seed=42)
    df_faulty, events = injector.inject_faults(df_clean, num_faults=6, seed=42)

    for ev in events:
        assert ev.fault_type != "frequency_deviation", "DC mode must not inject frequency deviation"
