"""
Isolation Forest Machine-Learning Anomaly Detection Engine.

Wraps scikit-learn's Isolation Forest algorithm for unsupervised electrical anomaly
detection. Learns the multivariate topology of normal aircraft electrical operations
and computes continuous anomaly scores without ground-truth label leakage.

CONVENTION MAPPING:
  scikit-learn: +1 = inlier/normal, -1 = outlier/anomaly
  Project convention: 0 = Normal, 1 = Anomaly
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import AppConfig


class IsolationForestDetector:
    """Isolation Forest anomaly detector for electrical-bus telemetry.

    Parameters
    ----------
    config : AppConfig
        Application configuration containing Isolation Forest hyperparameters.
    logger : logging.Logger, optional
        Logger instance.
    """

    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        cfg = self.config.isolation_forest
        self.model = IsolationForest(
            n_estimators=cfg.n_estimators,
            contamination=cfg.contamination,
            max_samples=cfg.max_samples,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.score_min: float = -0.5
        self.score_max: float = 0.5

    def fit(self, X_normal: pd.DataFrame) -> IsolationForestDetector:
        """Fit the Isolation Forest model strictly on normal baseline operating data.

        Parameters
        ----------
        X_normal : pd.DataFrame
            Feature matrix consisting purely of normal operating observations.
            Must NOT contain ground-truth columns.

        Returns
        -------
        IsolationForestDetector
            Fitted detector instance.
        """
        self.logger.info(
            f"Fitting Isolation Forest on {len(X_normal)} baseline normal samples "
            f"using {X_normal.shape[1]} features (n_estimators={self.model.n_estimators}, "
            f"contamination={self.model.contamination})."
        )
        self.model.fit(X_normal)
        self.is_fitted = True

        # Compute baseline decision score bounds for normalization
        scores = self.model.decision_function(X_normal)
        self.score_min = float(np.min(scores))
        self.score_max = float(np.max(scores))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies and compute anomaly scores for the input feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Scaled feature matrix.

        Returns
        -------
        pd.DataFrame
            DataFrame containing:
            - 'ml_raw_score': float (raw decision_function output, lower = more abnormal)
            - 'ml_anomaly_score': float (normalized [0, 1] where 1.0 = most anomalous)
            - 'ml_anomaly': int (0 = normal, 1 = anomaly according to model offset)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Isolation Forest model has not been fitted yet. Call fit() before predict()."
            )

        # Raw decision function: positive = inlier, negative = outlier
        raw_scores = self.model.decision_function(X)

        # Sklearn predict: +1 for normal, -1 for anomaly
        raw_preds = self.model.predict(X)

        # Map to project standard: 0 = Normal, 1 = Anomaly
        ml_anomaly = np.where(raw_preds == -1, 1, 0)

        # Normalize score into [0.0, 1.0] where 1.0 represents highest anomaly probability
        # Invert so higher values mean more anomalous
        span = max(self.score_max - self.score_min, 1e-6)
        # Shift and scale relative to baseline distribution
        norm_scores = (self.score_max - raw_scores) / span
        norm_scores = np.clip(norm_scores, 0.0, 1.0)

        results = pd.DataFrame(
            {
                "ml_raw_score": np.round(raw_scores, 4),
                "ml_anomaly_score": np.round(norm_scores, 4),
                "ml_anomaly": ml_anomaly,
            },
            index=X.index,
        )

        detected = int(np.sum(ml_anomaly))
        self.logger.info(
            f"Isolation Forest inference complete: {detected}/{len(X)} observations flagged as anomalies."
        )
        return results
