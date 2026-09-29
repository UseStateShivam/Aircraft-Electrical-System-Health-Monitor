"""
Visualization and Engineering Dashboard Generation Module.

Generates high-resolution publication-quality diagnostic plots for time-series sensor
data, anomaly distributions, fault intervals, Isolation Forest scores, and executive summary dashboards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import AppConfig
from src.evaluation import ClassificationMetrics, EventEvaluationSummary
from src.fault_injection import FaultEvent
from src.utils import ensure_directories


class Visualizer:
    """Generates time-series and diagnostic figures for the health monitor.

    Parameters
    ----------
    config : AppConfig
        Application configuration containing nominal values and thresholds.
    logger : logging.Logger, optional
        Logger instance.
    """

    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Set consistent engineering styling
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
        plt.rcParams["axes.edgecolor"] = "#4A5568"
        plt.rcParams["axes.linewidth"] = 0.8

    def plot_all(
        self,
        df_master: pd.DataFrame,
        events: List[FaultEvent],
        metrics: ClassificationMetrics,
        fault_df: pd.DataFrame,
        event_summary: EventEvaluationSummary,
        output_dir: str | Path = "outputs/figures",
    ) -> List[Path]:
        """Generate and save all primary plots and executive dashboard."""
        out_path = Path(output_dir)
        ensure_directories(out_path)

        bus_type = str(df_master["bus_type"].iloc[0]).lower()
        saved_files: List[Path] = []

        # 1. Voltage Timeseries
        p1 = self.plot_voltage_timeseries(df_master, events, out_path / "voltage_timeseries.png")
        saved_files.append(p1)

        # 2. Current Timeseries
        p2 = self.plot_current_timeseries(df_master, events, out_path / "current_timeseries.png")
        saved_files.append(p2)

        # 3. Frequency Timeseries (if AC)
        if bus_type == "ac":
            p3 = self.plot_frequency_timeseries(df_master, events, out_path / "frequency_timeseries.png")
            saved_files.append(p3)

        # 4. Detected Anomalies vs Ground Truth
        p4 = self.plot_detected_anomalies(df_master, events, out_path / "detected_anomalies.png")
        saved_files.append(p4)

        # 5. Isolation Forest Anomaly Score
        p5 = self.plot_anomaly_score(df_master, events, out_path / "anomaly_score.png")
        saved_files.append(p5)

        # 6. Executive Engineering Dashboard
        p6 = self.plot_system_dashboard(
            df_master, metrics, fault_df, event_summary, out_path / "system_summary.png"
        )
        saved_files.append(p6)

        return saved_files

    def _shade_fault_intervals(self, ax: plt.Axes, events: List[FaultEvent]) -> None:
        """Helper to draw vertical spans over injected fault intervals."""
        for ev in events:
            ax.axvspan(
                ev.start_time,
                ev.end_time,
                color="#EF4444",
                alpha=0.18,
                label="Injected Fault Event" if ev == events[0] else "",
            )

    def plot_voltage_timeseries(
        self, df: pd.DataFrame, events: List[FaultEvent], save_path: Path
    ) -> Path:
        """Plot Voltage vs Time with fault intervals and threshold bounds."""
        bus_type = str(df["bus_type"].iloc[0]).lower()
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)

        t = df["timestamp"].values
        v = df["voltage"].values

        ax.plot(t, v, label="Measured Voltage", color="#1E3A8A", linewidth=1.2)
        self._shade_fault_intervals(ax, events)

        # Draw thresholds
        if bus_type == "ac":
            th = self.config.thresholds_ac
            ax.axhline(th.voltage_warning_high, color="#F59E0B", linestyle="--", label="Warning Limits")
            ax.axhline(th.voltage_warning_low, color="#F59E0B", linestyle="--")
            ax.axhline(th.voltage_critical_high, color="#DC2626", linestyle=":", label="Critical Limits")
            ax.axhline(th.voltage_critical_low, color="#DC2626", linestyle=":")
            ax.set_ylabel("Voltage [V RMS]")
            ax.set_title("Aircraft AC Bus Voltage Telemetry (115V RMS Nominal)")
        else:
            th_dc = self.config.thresholds_dc
            ax.axhline(th_dc.voltage_warning_high, color="#F59E0B", linestyle="--", label="Warning Limits")
            ax.axhline(th_dc.voltage_warning_low, color="#F59E0B", linestyle="--")
            ax.axhline(th_dc.voltage_critical_high, color="#DC2626", linestyle=":", label="Critical Limits")
            ax.axhline(th_dc.voltage_critical_low, color="#DC2626", linestyle=":")
            ax.set_ylabel("Voltage [V DC]")
            ax.set_title("Aircraft DC Bus Voltage Telemetry (28V DC Nominal)")

        ax.set_xlabel("Time [seconds]")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_current_timeseries(
        self, df: pd.DataFrame, events: List[FaultEvent], save_path: Path
    ) -> Path:
        """Plot Bus Current vs Time with operating load profile."""
        bus_type = str(df["bus_type"].iloc[0]).lower()
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)

        t = df["timestamp"].values
        c = df["current"].values

        ax.plot(t, c, label="Measured Current", color="#047857", linewidth=1.2)
        self._shade_fault_intervals(ax, events)

        if bus_type == "ac":
            th = self.config.thresholds_ac
            ax.axhline(th.current_warning_high, color="#F59E0B", linestyle="--", label="Warning Limit")
            ax.axhline(th.current_critical_high, color="#DC2626", linestyle=":", label="Critical Limit")
        else:
            th_dc = self.config.thresholds_dc
            ax.axhline(th_dc.current_warning_high, color="#F59E0B", linestyle="--", label="Warning Limit")
            ax.axhline(th_dc.current_critical_high, color="#DC2626", linestyle=":", label="Critical Limit")

        ax.set_xlabel("Time [seconds]")
        ax.set_ylabel("Current [Amperes]")
        ax.set_title(f"Aircraft {bus_type.upper()} Bus Load Current Telemetry")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_frequency_timeseries(
        self, df: pd.DataFrame, events: List[FaultEvent], save_path: Path
    ) -> Path:
        """Plot Frequency vs Time with frequency regulation envelopes (AC bus)."""
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)

        t = df["timestamp"].values
        f = df["frequency"].values

        ax.plot(t, f, label="Measured Frequency", color="#6D28D9", linewidth=1.2)
        self._shade_fault_intervals(ax, events)

        th = self.config.thresholds_ac
        ax.axhline(th.frequency_warning_high, color="#F59E0B", linestyle="--", label="Warning Limits")
        ax.axhline(th.frequency_warning_low, color="#F59E0B", linestyle="--")
        ax.axhline(th.frequency_critical_high, color="#DC2626", linestyle=":", label="Critical Limits")
        ax.axhline(th.frequency_critical_low, color="#DC2626", linestyle=":")

        ax.set_xlabel("Time [seconds]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title("Aircraft AC Bus Frequency Telemetry (400 Hz Nominal)")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_detected_anomalies(
        self, df: pd.DataFrame, events: List[FaultEvent], save_path: Path
    ) -> Path:
        """Plot dual panel comparing ground-truth faults vs system-detected anomalies."""
        bus_type = str(df["bus_type"].iloc[0]).lower()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, dpi=150)

        t = df["timestamp"].values
        v = df["voltage"].values
        anom_mask = df["final_anomaly"] == 1
        true_mask = df["true_fault"] == 1

        # Panel 1: Voltage with detections
        ax1.plot(t, v, color="#475569", alpha=0.6, label="Voltage Signal")
        ax1.scatter(
            t[anom_mask],
            v[anom_mask],
            color="#DC2626",
            s=18,
            label="Detected Anomaly",
            zorder=4,
        )
        self._shade_fault_intervals(ax1, events)
        ax1.set_ylabel(f"Voltage [{ 'V RMS' if bus_type == 'ac' else 'V DC' }]")
        ax1.set_title("System-Detected Anomalies Highlighted on Voltage Telemetry")
        ax1.legend(loc="upper right", framealpha=0.9)

        # Panel 2: Ground Truth vs Detected Flag
        ax2.step(t, df["true_fault"], where="mid", color="#2563EB", label="Ground Truth (1=Fault)", linewidth=1.5)
        ax2.step(
            t,
            df["final_anomaly"] * 0.9,
            where="mid",
            color="#DC2626",
            linestyle="--",
            label="Detected by System (1=Anomaly)",
            linewidth=1.2,
        )
        ax2.set_xlabel("Time [seconds]")
        ax2.set_ylabel("Binary State")
        ax2.set_ylim(-0.1, 1.2)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["Normal", "Anomaly"])
        ax2.set_title("Ground-Truth Fault Injection Interval vs Pipeline Prediction")
        ax2.legend(loc="upper right", framealpha=0.9)

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_anomaly_score(
        self, df: pd.DataFrame, events: List[FaultEvent], save_path: Path
    ) -> Path:
        """Plot Isolation Forest continuous anomaly score profile over time."""
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)

        t = df["timestamp"].values
        score = df["ml_anomaly_score"].values

        ax.plot(t, score, color="#7C3AED", label="Isolation Forest Normalized Score", linewidth=1.2)
        self._shade_fault_intervals(ax, events)

        # Threshold line
        ax.axhline(0.50, color="#DC2626", linestyle="--", label="Model Inlier/Outlier Frontier")
        ax.set_xlabel("Time [seconds]")
        ax.set_ylabel("Anomaly Probability Score [0 - 1]")
        ax.set_title("Unsupervised Isolation Forest Continuous Anomaly Score")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", framealpha=0.9)

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_system_dashboard(
        self,
        df: pd.DataFrame,
        metrics: ClassificationMetrics,
        fault_df: pd.DataFrame,
        event_summary: EventEvaluationSummary,
        save_path: Path,
    ) -> Path:
        """Generate multi-panel summary engineering dashboard."""
        bus_type = str(df["bus_type"].iloc[0]).upper()
        fig = plt.figure(figsize=(14, 9), dpi=160)
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

        # Panel 1: Health Score over Time
        ax1 = fig.add_subplot(gs[0, 0])
        t = df["timestamp"].values
        h_score = df["health_score"].values
        ax1.plot(t, h_score, color="#2563EB", linewidth=1.5, label="Bus Health Index")
        ax1.axhline(80, color="#F59E0B", linestyle=":", label="Warning Threshold (80)")
        ax1.axhline(50, color="#DC2626", linestyle="--", label="Critical Threshold (50)")
        ax1.set_xlabel("Time [seconds]")
        ax1.set_ylabel("Health Score (0 - 100)")
        ax1.set_title(f"Dynamic System Health Score ({bus_type} Bus)")
        ax1.set_ylim(0, 105)
        ax1.legend(loc="lower left", fontsize=8)

        # Panel 2: Confusion Matrix Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        cm = np.array(
            [[metrics.true_negatives, metrics.false_positives], [metrics.false_negatives, metrics.true_positives]]
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax2,
            xticklabels=["Pred Normal", "Pred Anomaly"],
            yticklabels=["True Normal", "True Fault"],
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax2.set_title(
            f"Confusion Matrix (Acc: {metrics.accuracy*100:.1f}%, F1: {metrics.f1*100:.1f}%)"
        )

        # Panel 3: Fault Detection Rates by Type
        ax3 = fig.add_subplot(gs[1, 0])
        # Filter out normal baseline row
        f_eval = fault_df[~fault_df["fault_type"].str.contains("normal")].copy()
        if not f_eval.empty:
            colors = ["#10B981" if r >= 80 else "#F59E0B" if r >= 50 else "#EF4444" for r in f_eval["detection_rate"]]
            bars = ax3.barh(f_eval["fault_type"], f_eval["detection_rate"], color=colors, height=0.65)
            ax3.set_xlim(0, 110)
            ax3.set_xlabel("Detection Rate [%]")
            ax3.set_title("Detection Rate by Fault Category")
            for bar in bars:
                w = bar.get_width()
                ax3.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8)
        else:
            ax3.text(0.5, 0.5, "No specific fault breakdown available", ha="center")

        # Panel 4: Health Status Distribution & Key Performance Indicators
        ax4 = fig.add_subplot(gs[1, 1])
        status_counts = df["health_status"].value_counts()
        palette_map = {
            "HEALTHY": "#10B981",
            "WARNING": "#F59E0B",
            "DEGRADED": "#8B5CF6",
            "CRITICAL": "#EF4444",
            "SENSOR_FAULT": "#EC4899",
        }
        pie_colors = [palette_map.get(s, "#6B7280") for s in status_counts.index]
        ax4.pie(
            status_counts.values,
            labels=status_counts.index,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 9},
        )
        ax4.set_title(
            f"Operating Status Distribution\n(Event Recall: {event_summary.event_detection_rate:.1f}%, Latency: {event_summary.mean_detection_delay_sec:.2f}s)"
        )

        fig.suptitle(
            f"Aircraft Electrical Bus Health Monitor - Engineering Dashboard ({bus_type})",
            fontsize=15,
            weight="bold",
            y=0.98,
        )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95)
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
