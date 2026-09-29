"""
Synthetic Aircraft Electrical-Bus Sensor Data Simulator.

Generates realistic time-series sensor observations representing normal aircraft
electrical-bus operation under varying operating conditions (low load, normal load,
high load, transient load).

Supports both:
  - 115 V AC RMS, ~400 Hz electrical bus
  - 28 V DC electrical bus

DISCLAIMER: Educational simulation/prototype values only. Not certified avionics specifications.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import AppConfig


class ElectricalBusSimulator:
    """Simulator for aircraft electrical bus time-series sensor data.

    Parameters
    ----------
    config : AppConfig
        Application configuration containing bus parameters, operating conditions,
        and noise levels.
    logger : logging.Logger, optional
        Logger instance for tracing simulation steps.
    """

    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def generate(
        self,
        duration_seconds: Optional[float] = None,
        sampling_rate_hz: Optional[float] = None,
        bus_type: Optional[str] = None,
        noise_scenario: str = "normal_noise",
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic normal electrical operating data.

        Parameters
        ----------
        duration_seconds : float, optional
            Duration of the simulation in seconds. Defaults to config value.
        sampling_rate_hz : float, optional
            Sampling frequency in Hz. Defaults to config value.
        bus_type : str, optional
            Bus mode: 'ac' or 'dc'. Defaults to config value.
        noise_scenario : str
            Noise intensity multiplier profile: 'low_noise', 'normal_noise', 'high_noise'.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame containing time-series data with timestamps, electrical signals,
            operating condition, and initial ground-truth columns.
        """
        dur = float(duration_seconds or self.config.duration_seconds)
        fs = float(sampling_rate_hz or self.config.sampling_rate_hz)
        mode = (bus_type or self.config.bus_type).lower()
        active_seed = seed if seed is not None else self.config.seed

        rng = np.random.RandomState(active_seed)

        n_samples = int(np.round(dur * fs))
        timestamps = np.linspace(0.0, dur, n_samples, endpoint=False)

        noise_mult = float(self.config.noise_scenarios.get(noise_scenario, 1.0))

        self.logger.info(
            f"Generating {n_samples} samples for {mode.upper()} bus "
            f"({dur:.1f}s @ {fs:.1f}Hz, noise='{noise_scenario}')"
        )

        # 1. Build operating condition sequence
        op_conditions, raw_load = self._generate_operating_profile(
            n_samples=n_samples, fs=fs, rng=rng
        )

        # Smooth load level to emulate thermal/mechanical inertia
        load_level = self._smooth_signal(raw_load, window_size=int(fs * 2))
        load_level = np.clip(load_level, 0.20, 1.30)

        # 2. Synthesize electrical signals based on bus mode
        if mode == "ac":
            df = self._generate_ac_signals(
                timestamps=timestamps,
                load_level=load_level,
                op_conditions=op_conditions,
                noise_mult=noise_mult,
                rng=rng,
            )
        elif mode == "dc":
            df = self._generate_dc_signals(
                timestamps=timestamps,
                load_level=load_level,
                op_conditions=op_conditions,
                noise_mult=noise_mult,
                rng=rng,
            )
        else:
            raise ValueError(f"Unsupported bus_type: '{mode}'. Must be 'ac' or 'dc'.")

        return df

    def _generate_operating_profile(
        self, n_samples: int, fs: float, rng: np.random.RandomState
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct realistic operational profile transitions across the time span."""
        conditions_cfg = self.config.operating_conditions.get(
            "conditions",
            [
                {"name": "low_load", "load_multiplier": 0.35, "duration_range": [40, 80]},
                {"name": "normal_load", "load_multiplier": 0.65, "duration_range": [60, 120]},
                {"name": "high_load", "load_multiplier": 0.95, "duration_range": [40, 90]},
                {"name": "transient_load", "load_multiplier": 1.15, "duration_range": [5, 15]},
            ],
        )

        op_names: list[str] = []
        load_series: list[float] = []

        current_sample = 0
        cond_idx = 0
        n_conds = len(conditions_cfg)

        while current_sample < n_samples:
            cond = conditions_cfg[cond_idx % n_conds]
            d_min, d_max = cond.get("duration_range", [30, 90])
            dur_sec = rng.uniform(d_min, d_max)
            samples_in_segment = int(np.round(dur_sec * fs))
            remaining = n_samples - current_sample
            chunk_len = min(samples_in_segment, remaining)

            base_load = float(cond.get("load_multiplier", 0.65))
            # Slight random wander within the segment
            wander = rng.normal(0.0, 0.02, size=chunk_len)
            segment_load = base_load + wander

            op_names.extend([cond["name"]] * chunk_len)
            load_series.extend(segment_load.tolist())

            current_sample += chunk_len
            cond_idx += 1

        return np.array(op_names), np.array(load_series)

    def _smooth_signal(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """Apply simple moving average smoothing to mimic system physical inertia."""
        if window_size <= 1:
            return signal
        kernel = np.ones(window_size) / window_size
        padded = np.pad(signal, (window_size // 2, window_size - 1 - window_size // 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed

    def _generate_ac_signals(
        self,
        timestamps: np.ndarray,
        load_level: np.ndarray,
        op_conditions: np.ndarray,
        noise_mult: float,
        rng: np.random.RandomState,
    ) -> pd.DataFrame:
        """Synthesize 115 V AC 400 Hz electrical bus parameters."""
        p = self.config.ac_bus
        n_samples = len(timestamps)

        # Slow generator regulation droop under load: -1.5 V per unit load above 0.5
        voltage_droop = -1.8 * (load_level - 0.5)

        # Slow thermal drift / sub-harmonic oscillation (0.03 Hz)
        slow_voltage_drift = 0.4 * np.sin(2 * np.pi * 0.03 * timestamps)

        # Measurement sensor noise
        voltage_noise = rng.normal(0.0, p.voltage_noise_std * noise_mult, size=n_samples)
        voltage = p.nominal_voltage + voltage_droop + slow_voltage_drift + voltage_noise

        # Current strongly correlates with load level
        current_noise = rng.normal(0.0, p.current_noise_std * noise_mult, size=n_samples)
        current = p.base_current * load_level + current_noise
        current = np.maximum(current, 0.5)  # Physical constraint: non-negative current

        # Frequency: 400 Hz nominal with slight governor drift (0.015 Hz) & noise
        freq_drift = 0.15 * np.cos(2 * np.pi * 0.015 * timestamps + 0.4)
        # Slight load-frequency coupling (minor droop on high load)
        freq_droop = -0.12 * (load_level - 0.5)
        freq_noise = rng.normal(0.0, p.frequency_noise_std * noise_mult, size=n_samples)
        frequency = p.nominal_frequency + freq_drift + freq_droop + freq_noise

        apparent_power = voltage * current

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "bus_type": "ac",
                "operating_condition": op_conditions,
                "load_level": np.round(load_level, 4),
                "voltage": np.round(voltage, 3),
                "current": np.round(current, 3),
                "frequency": np.round(frequency, 3),
                "apparent_power": np.round(apparent_power, 2),
                "true_fault": 0,
                "fault_type": "normal",
                "fault_id": "NONE",
                "severity": "none",
            }
        )
        return df

    def _generate_dc_signals(
        self,
        timestamps: np.ndarray,
        load_level: np.ndarray,
        op_conditions: np.ndarray,
        noise_mult: float,
        rng: np.random.RandomState,
    ) -> pd.DataFrame:
        """Synthesize 28 V DC electrical bus parameters (no AC frequency)."""
        p = self.config.dc_bus
        n_samples = len(timestamps)

        # DC bus voltage droop with load: -0.8 V per unit load
        voltage_droop = -0.75 * (load_level - 0.5)
        slow_voltage_drift = 0.12 * np.sin(2 * np.pi * 0.02 * timestamps)
        voltage_noise = rng.normal(0.0, p.voltage_noise_std * noise_mult, size=n_samples)
        voltage = p.nominal_voltage + voltage_droop + slow_voltage_drift + voltage_noise

        # DC current
        current_noise = rng.normal(0.0, p.current_noise_std * noise_mult, size=n_samples)
        current = p.base_current * load_level + current_noise
        current = np.maximum(current, 1.0)

        # DC buses have no AC frequency -> explicitly np.nan
        frequency = np.full(n_samples, np.nan)

        apparent_power = voltage * current

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "bus_type": "dc",
                "operating_condition": op_conditions,
                "load_level": np.round(load_level, 4),
                "voltage": np.round(voltage, 3),
                "current": np.round(current, 3),
                "frequency": frequency,
                "apparent_power": np.round(apparent_power, 2),
                "true_fault": 0,
                "fault_type": "normal",
                "fault_id": "NONE",
                "severity": "none",
            }
        )
        return df
