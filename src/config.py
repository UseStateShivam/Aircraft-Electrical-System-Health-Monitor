"""
Configuration loading and validation for the Aircraft Electrical System Health Monitor.

Loads YAML configuration files and enforces integrity constraints on sampling,
electrical bus specifications, thresholds, and fault injection parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


class ConfigValidationError(ValueError):
    """Exception raised when configuration parameters fail validation rules."""


@dataclass
class ACBusConfig:
    nominal_voltage: float = 115.0
    nominal_frequency: float = 400.0
    voltage_noise_std: float = 0.7
    frequency_noise_std: float = 0.25
    base_current: float = 20.0
    current_noise_std: float = 0.35
    power_factor: float = 0.92


@dataclass
class DCBusConfig:
    nominal_voltage: float = 28.0
    voltage_noise_std: float = 0.22
    base_current: float = 50.0
    current_noise_std: float = 0.60


@dataclass
class ACThresholds:
    voltage_critical_low: float = 100.0
    voltage_warning_low: float = 108.0
    voltage_warning_high: float = 122.0
    voltage_critical_high: float = 130.0
    current_warning_high: float = 28.0
    current_critical_high: float = 35.0
    frequency_critical_low: float = 385.0
    frequency_warning_low: float = 395.0
    frequency_warning_high: float = 405.0
    frequency_critical_high: float = 415.0


@dataclass
class DCThresholds:
    voltage_critical_low: float = 22.0
    voltage_warning_low: float = 25.0
    voltage_warning_high: float = 30.0
    voltage_critical_high: float = 32.5
    current_warning_high: float = 70.0
    current_critical_high: float = 90.0


@dataclass
class IsolationForestConfig:
    n_estimators: int = 150
    contamination: float = 0.05
    max_samples: Any = "auto"
    random_state: int = 42
    features_to_use: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class AppConfig:
    seed: int = 42
    bus_type: str = "ac"
    sampling_rate_hz: float = 10.0
    duration_seconds: float = 600.0
    ac_bus: ACBusConfig = field(default_factory=ACBusConfig)
    dc_bus: DCBusConfig = field(default_factory=DCBusConfig)
    thresholds_ac: ACThresholds = field(default_factory=ACThresholds)
    thresholds_dc: DCThresholds = field(default_factory=DCThresholds)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    fault_injection: Dict[str, Any] = field(default_factory=dict)
    operating_conditions: Dict[str, Any] = field(default_factory=dict)
    noise_scenarios: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    health_score: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    raw_dict: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str | Path = "config/config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    AppConfig
        Validated configuration dataclass instance.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ConfigValidationError
        If configuration parameters violate schema or engineering constraints.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found at '{path.resolve()}'. "
            "Please ensure config/config.yaml exists before executing the pipeline."
        )

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ConfigValidationError(f"Invalid YAML syntax in configuration file: {exc}") from exc

    return validate_config(data)


def validate_config(data: Dict[str, Any]) -> AppConfig:
    """Validate raw dictionary from YAML and construct AppConfig.

    Parameters
    ----------
    data : Dict[str, Any]
        Raw dictionary from parsed YAML.

    Returns
    -------
    AppConfig
        Structured and validated configuration.
    """
    bus_type = str(data.get("bus_type", "ac")).lower()
    if bus_type not in ("ac", "dc"):
        raise ConfigValidationError(
            f"Invalid bus_type '{bus_type}'. Must be either 'ac' or 'dc'."
        )

    sampling_rate = float(data.get("sampling_rate_hz", 10.0))
    if sampling_rate <= 0:
        raise ConfigValidationError(f"sampling_rate_hz must be > 0, got {sampling_rate}")

    duration = float(data.get("duration_seconds", 600.0))
    if duration <= 0:
        raise ConfigValidationError(f"duration_seconds must be > 0, got {duration}")

    seed = int(data.get("seed", 42))

    # Parse AC Bus
    ac_raw = data.get("ac_bus", {})
    ac_bus = ACBusConfig(
        nominal_voltage=float(ac_raw.get("nominal_voltage", 115.0)),
        nominal_frequency=float(ac_raw.get("nominal_frequency", 400.0)),
        voltage_noise_std=float(ac_raw.get("voltage_noise_std", 0.7)),
        frequency_noise_std=float(ac_raw.get("frequency_noise_std", 0.25)),
        base_current=float(ac_raw.get("base_current", 20.0)),
        current_noise_std=float(ac_raw.get("current_noise_std", 0.35)),
        power_factor=float(ac_raw.get("power_factor", 0.92)),
    )

    # Parse DC Bus
    dc_raw = data.get("dc_bus", {})
    dc_bus = DCBusConfig(
        nominal_voltage=float(dc_raw.get("nominal_voltage", 28.0)),
        voltage_noise_std=float(dc_raw.get("voltage_noise_std", 0.22)),
        base_current=float(dc_raw.get("base_current", 50.0)),
        current_noise_std=float(dc_raw.get("current_noise_std", 0.60)),
    )

    # Parse Thresholds
    thresh_raw = data.get("thresholds", {})
    thresh_ac_raw = thresh_raw.get("ac", {})
    thresh_ac = ACThresholds(
        voltage_critical_low=float(thresh_ac_raw.get("voltage_critical_low", 100.0)),
        voltage_warning_low=float(thresh_ac_raw.get("voltage_warning_low", 108.0)),
        voltage_warning_high=float(thresh_ac_raw.get("voltage_warning_high", 122.0)),
        voltage_critical_high=float(thresh_ac_raw.get("voltage_critical_high", 130.0)),
        current_warning_high=float(thresh_ac_raw.get("current_warning_high", 28.0)),
        current_critical_high=float(thresh_ac_raw.get("current_critical_high", 35.0)),
        frequency_critical_low=float(thresh_ac_raw.get("frequency_critical_low", 385.0)),
        frequency_warning_low=float(thresh_ac_raw.get("frequency_warning_low", 395.0)),
        frequency_warning_high=float(thresh_ac_raw.get("frequency_warning_high", 405.0)),
        frequency_critical_high=float(thresh_ac_raw.get("frequency_critical_high", 415.0)),
    )

    # Validate AC threshold order
    if not (
        thresh_ac.voltage_critical_low
        < thresh_ac.voltage_warning_low
        < thresh_ac.voltage_warning_high
        < thresh_ac.voltage_critical_high
    ):
        raise ConfigValidationError("AC voltage thresholds are not strictly monotonically increasing.")

    if thresh_ac.current_warning_high >= thresh_ac.current_critical_high:
        raise ConfigValidationError("AC current_warning_high must be less than current_critical_high.")

    if not (
        thresh_ac.frequency_critical_low
        < thresh_ac.frequency_warning_low
        < thresh_ac.frequency_warning_high
        < thresh_ac.frequency_critical_high
    ):
        raise ConfigValidationError("AC frequency thresholds are not strictly monotonically increasing.")

    thresh_dc_raw = thresh_raw.get("dc", {})
    thresh_dc = DCThresholds(
        voltage_critical_low=float(thresh_dc_raw.get("voltage_critical_low", 22.0)),
        voltage_warning_low=float(thresh_dc_raw.get("voltage_warning_low", 25.0)),
        voltage_warning_high=float(thresh_dc_raw.get("voltage_warning_high", 30.0)),
        voltage_critical_high=float(thresh_dc_raw.get("voltage_critical_high", 32.5)),
        current_warning_high=float(thresh_dc_raw.get("current_warning_high", 70.0)),
        current_critical_high=float(thresh_dc_raw.get("current_critical_high", 90.0)),
    )

    if not (
        thresh_dc.voltage_critical_low
        < thresh_dc.voltage_warning_low
        < thresh_dc.voltage_warning_high
        < thresh_dc.voltage_critical_high
    ):
        raise ConfigValidationError("DC voltage thresholds are not strictly monotonically increasing.")

    if thresh_dc.current_warning_high >= thresh_dc.current_critical_high:
        raise ConfigValidationError("DC current_warning_high must be less than current_critical_high.")

    # Isolation Forest
    if_raw = data.get("isolation_forest", {})
    contamination = float(if_raw.get("contamination", 0.05))
    if not (0.0 < contamination < 0.5):
        raise ConfigValidationError(f"contamination must be in (0, 0.5), got {contamination}")

    iso_forest = IsolationForestConfig(
        n_estimators=int(if_raw.get("n_estimators", 150)),
        contamination=contamination,
        max_samples=if_raw.get("max_samples", "auto"),
        random_state=int(if_raw.get("random_state", seed)),
        features_to_use=if_raw.get("features_to_use", {}),
    )

    return AppConfig(
        seed=seed,
        bus_type=bus_type,
        sampling_rate_hz=sampling_rate,
        duration_seconds=duration,
        ac_bus=ac_bus,
        dc_bus=dc_bus,
        thresholds_ac=thresh_ac,
        thresholds_dc=thresh_dc,
        isolation_forest=iso_forest,
        fault_injection=data.get("fault_injection", {}),
        operating_conditions=data.get("operating_conditions", {}),
        noise_scenarios=data.get("noise_scenarios", {}),
        features=data.get("features", {}),
        health_score=data.get("health_score", {}),
        paths=data.get("paths", {}),
        raw_dict=data,
    )
