"""
Unit tests for Isolation Forest Anomaly Detector.

Validates model fitting, score normalization [0, 1], prediction convention (0=normal, 1=anomaly),
and outlier sensitivity.
"""

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detector import IsolationForestDetector
from src.config import load_config


@pytest.fixture
def config():
    return load_config("config/config.yaml")


def test_fit_and_prediction_convention(config):
    """Test that Isolation Forest fits and produces project standard predictions."""
    detector = IsolationForestDetector(config=config)

    # 500 normal training points around mean 0
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(
        rng.normal(0, 1, size=(500, 4)),
        columns=["f1", "f2", "f3", "f4"],
    )

    detector.fit(X_train)
    assert detector.is_fitted

    # Test on a mixture of normal and extreme outlier points
    X_test = pd.DataFrame(
        [
            [0.1, -0.2, 0.05, 0.1],  # Normal
            [25.0, -30.0, 40.0, 50.0],  # Obvious extreme anomaly
        ],
        columns=["f1", "f2", "f3", "f4"],
    )

    res = detector.predict(X_test)

    assert "ml_anomaly" in res.columns
    assert "ml_anomaly_score" in res.columns
    assert "ml_raw_score" in res.columns

    # Check conventions: 0 = Normal, 1 = Anomaly
    assert res.loc[0, "ml_anomaly"] == 0
    assert res.loc[1, "ml_anomaly"] == 1

    # Check bounds on normalized score: in [0, 1]
    assert (res["ml_anomaly_score"] >= 0.0).all()
    assert (res["ml_anomaly_score"] <= 1.0).all()

    # Outlier should have higher anomaly score than normal
    assert res.loc[1, "ml_anomaly_score"] > res.loc[0, "ml_anomaly_score"]
