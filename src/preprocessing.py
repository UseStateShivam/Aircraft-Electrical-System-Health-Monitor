"""
Data Preprocessing and Feature Engineering Module.

Handles data validation, sensor dropout detection, physically impossible value checks,
missing-value imputation for ML compatibility, feature engineering (deviations, rates,
rolling statistics), and standard scaling.

STRICT DATA LEAKAGE POLICY:
Ground-truth columns (true_fault, fault_type, fault_id, severity) are strictly quarantined
and NEVER included in feature matrices fed to machine learning models.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import AppConfig


class PreprocessingValidationError(ValueError):
    """Exception raised when dataset structure or electrical values fail validation."""


class DataPreprocessor:
    """Preprocesses electrical time-series data and constructs engineering features.

    Parameters
    ----------
    config : AppConfig
        Application configuration containing bus parameters and feature settings.
    logger : logging.Logger, optional
        Logger instance for tracing preprocessing operations.
    """

    REQUIRED_COLUMNS = [
        "timestamp",
        "bus_type",
        "operating_condition",
        "voltage",
        "current",
        "frequency",
    ]

    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """Verify that the dataset contains all required columns and valid bus types.

        Raises
        ------
        PreprocessingValidationError
            If required columns are missing or if bus_type is unrecognized.
        """
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise PreprocessingValidationError(
                f"Dataset is missing required columns: {missing_cols}"
            )

        bus_type = str(df["bus_type"].iloc[0]).lower()
        if bus_type not in ("ac", "dc"):
            raise PreprocessingValidationError(
                f"Unsupported bus_type '{bus_type}'. Expected 'ac' or 'dc'."
            )

    def process(
        self,
        df: pd.DataFrame,
        is_training: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess electrical sensor dataset and construct clean feature matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or fault-injected electrical sensor dataset.
        is_training : bool
            Whether this dataset represents normal training data (fits scaler if True).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            1. Enriched DataFrame with sensor quality flags and engineered features.
            2. Scaled feature DataFrame (X) ready for Isolation Forest.
        """
        self.validate_dataset(df)
        data = df.copy()
        bus_type = str(data["bus_type"].iloc[0]).lower()

        # 1. Sensor dropout / missing value detection
        # Check if voltage or current has NaN
        voltage_nan = data["voltage"].isna()
        current_nan = data["current"].isna()
        freq_nan = data["frequency"].isna() if bus_type == "ac" else pd.Series(False, index=data.index)

        data["sensor_dropout_flag"] = (voltage_nan | current_nan | freq_nan).astype(int)

        # 2. Impute missing values for feature computation (forward fill then backward fill)
        clean_voltage = data["voltage"].ffill().bfill()
        clean_current = data["current"].ffill().bfill()
        if bus_type == "ac":
            clean_frequency = data["frequency"].ffill().bfill()
        else:
            clean_frequency = pd.Series(0.0, index=data.index)

        # 3. Detect physically impossible sensor values (negative voltage/current)
        data["impossible_value_flag"] = (
            (clean_voltage < 0.0) | (clean_current < 0.0)
        ).astype(int)

        # 4. Feature Engineering
        dt = 1.0 / self.config.sampling_rate_hz
        nominal_v = (
            self.config.ac_bus.nominal_voltage
            if bus_type == "ac"
            else self.config.dc_bus.nominal_voltage
        )

        data["voltage_clean"] = clean_voltage
        data["current_clean"] = clean_current
        data["voltage_deviation"] = clean_voltage - nominal_v

        # Finite difference rates of change (d/dt)
        data["voltage_rate"] = clean_voltage.diff().fillna(0.0) / dt
        data["current_rate"] = clean_current.diff().fillna(0.0) / dt

        # Rolling statistics (short window e.g. 5 samples = 0.5s, longer window 20 = 2s)
        w_short = 5
        w_long = 20
        data["rolling_voltage_mean"] = clean_voltage.rolling(window=w_short, min_periods=1).mean()
        data["rolling_voltage_std"] = clean_voltage.rolling(window=w_long, min_periods=1).std().fillna(0.0)
        data["rolling_current_mean"] = clean_current.rolling(window=w_short, min_periods=1).mean()
        data["rolling_current_std"] = clean_current.rolling(window=w_long, min_periods=1).std().fillna(0.0)

        data["apparent_power"] = clean_voltage * clean_current

        if bus_type == "ac":
            nominal_f = self.config.ac_bus.nominal_frequency
            data["frequency_clean"] = clean_frequency
            data["frequency_deviation"] = clean_frequency - nominal_f
            data["frequency_rate"] = clean_frequency.diff().fillna(0.0) / dt
            data["rolling_frequency_mean"] = (
                clean_frequency.rolling(window=w_short, min_periods=1).mean()
            )
            data["rolling_frequency_std"] = (
                clean_frequency.rolling(window=w_long, min_periods=1).std().fillna(0.0)
            )

        # 5. Extract Feature Matrix (X) strictly excluding ground-truth columns
        feature_cols = self._get_feature_columns(bus_type)
        self.feature_names = feature_cols
        X_raw = data[feature_cols].copy()

        # Handle any residual NaN in engineered features
        X_raw = X_raw.ffill().bfill().fillna(0.0)

        # 6. Fit / Transform Scaler
        if is_training:
            X_scaled_arr = self.scaler.fit_transform(X_raw)
            self.is_fitted = True
            self.logger.info(
                f"Fitted standard scaler on training features: {feature_cols}"
            )
        else:
            if not self.is_fitted:
                # If evaluating without prior fit, fit on provided data with warning
                self.logger.warning(
                    "StandardScaler was not pre-fitted; fitting on current dataset."
                )
                X_scaled_arr = self.scaler.fit_transform(X_raw)
                self.is_fitted = True
            else:
                X_scaled_arr = self.scaler.transform(X_raw)

        X_scaled = pd.DataFrame(X_scaled_arr, columns=feature_cols, index=data.index)
        return data, X_scaled

    def _get_feature_columns(self, bus_type: str) -> List[str]:
        """Return the list of features to use based on configuration and bus type."""
        configured = self.config.isolation_forest.features_to_use.get(bus_type, [])
        if configured:
            return configured

        # Default fallbacks
        if bus_type == "ac":
            return [
                "voltage_clean",
                "current_clean",
                "frequency_clean",
                "voltage_deviation",
                "frequency_deviation",
                "voltage_rate",
                "current_rate",
                "rolling_voltage_mean",
                "rolling_voltage_std",
                "apparent_power",
            ]
        else:
            return [
                "voltage_clean",
                "current_clean",
                "voltage_deviation",
                "voltage_rate",
                "current_rate",
                "rolling_voltage_mean",
                "rolling_voltage_std",
                "apparent_power",
            ]
