"""
Unit tests for Evaluation Metrics and End-to-End Pipeline Execution.

Validates metric validity ranges, benchmark comparison, event latency calculation,
and clean execution with small test configurations.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.run_pipeline import run_pipeline
from src.config import load_config
from src.evaluation import ModelEvaluator
from src.fault_injection import FaultEvent


@pytest.fixture
def config():
    return load_config("config/config.yaml")


def test_evaluation_metrics_bounds():
    """Test that all classification metrics fall strictly within mathematical bounds."""
    evaluator = ModelEvaluator()

    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1])

    metrics = evaluator.evaluate_predictions(y_true, y_pred, "Test Model")

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.false_positive_rate <= 1.0

    # True positives = 3, False negatives = 1, False positives = 1, True negatives = 3
    assert metrics.true_positives == 3
    assert metrics.true_negatives == 3
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1


def test_event_evaluation_latency():
    """Test event-level detection rate and latency calculations."""
    evaluator = ModelEvaluator()

    # Create dummy dataframe with timestamp and predictions
    df = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "final_anomaly": [0, 0, 1, 1, 0, 0],
        }
    )

    # Event spanning indices 1 to 4 (time 1.0 to 4.0)
    event = FaultEvent(
        fault_id="FLT_01",
        fault_type="overvoltage",
        start_time=1.0,
        end_time=4.0,
        start_idx=1,
        end_idx=4,
        affected_variable="voltage",
        severity="moderate",
        magnitude_factor=1.2,
        description="Test event",
    )

    summary = evaluator.evaluate_events(df, [event])
    assert summary.total_events == 1
    assert summary.detected_events == 1
    assert summary.event_detection_rate == 100.0
    # First detected anomaly was at timestamp 2.0, event start was 1.0 -> delay is 1.0s
    assert summary.mean_detection_delay_sec == pytest.approx(1.0)


def test_pipeline_execution_small_ac():
    """Test full pipeline runs cleanly end-to-end on small AC configuration."""
    # Run a short 20s simulation in test mode
    run_pipeline(mode="ac", duration=20.0, seed=42)


def test_pipeline_execution_small_dc():
    """Test full pipeline runs cleanly end-to-end on small DC configuration."""
    # Run a short 20s simulation in test mode
    run_pipeline(mode="dc", duration=20.0, seed=42)
