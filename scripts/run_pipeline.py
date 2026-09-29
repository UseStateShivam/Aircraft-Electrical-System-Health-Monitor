"""
Aircraft Electrical System Health Monitor - Primary Execution Pipeline.

Executes the complete end-to-end workflow:
  1. Synthetic electrical data generation (AC/DC mode)
  2. Data preprocessing & feature engineering (strictly zero leakage)
  3. Controlled fault injection (10 electrical and sensor fault types)
  4. Engineering threshold monitoring
  5. Isolation Forest machine-learning anomaly detection
  6. Multi-tier health assessment & continuous health scoring
  7. Quantitative performance evaluation (sample, fault-type, and event scopes)
  8. Publication-quality diagnostic visualizations and executive dashboard
  9. Structured alert logs (CSV, JSON) and human-readable engineering report

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --mode dc
  python scripts/run_pipeline.py --mode ac --duration 600 --seed 42
  python scripts/run_pipeline.py --stream
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# Add project root to sys.path so src imports work reliably
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.anomaly_detector import IsolationForestDetector
from src.config import load_config
from src.evaluation import ModelEvaluator
from src.fault_injection import FaultInjector
from src.health_monitor import IntegratedHealthMonitor
from src.preprocessing import DataPreprocessor
from src.reporting import ReportGenerator
from src.simulator import ElectricalBusSimulator
from src.threshold_monitor import EngineeringThresholdMonitor
from src.utils import ensure_directories, set_random_seed, setup_logger
from src.visualization import Visualizer


def run_pipeline(
    mode: str = "ac",
    duration: float = 600.0,
    seed: int = 42,
    config_path: str = "config/config.yaml",
    is_stream_demo: bool = False,
) -> None:
    """Execute the full end-to-end health monitoring pipeline."""
    # 1. Setup logging and directory structure
    logger = setup_logger(log_file="outputs/reports/execution.log")
    set_random_seed(seed)

    ensure_directories(
        "data/raw",
        "data/processed",
        "data/reports",
        "outputs/figures",
        "outputs/reports",
        "outputs/metrics",
    )

    # 2. Load and validate configuration
    config = load_config(config_path)
    config.seed = seed
    config.duration_seconds = duration
    config.bus_type = mode.lower()

    bus_mode = config.bus_type.upper()
    nom_v = (
        f"{config.ac_bus.nominal_voltage:.0f} V RMS"
        if bus_mode == "AC"
        else f"{config.dc_bus.nominal_voltage:.0f} V DC"
    )
    nom_f = (
        f"{config.ac_bus.nominal_frequency:.0f} Hz"
        if bus_mode == "AC"
        else "N/A (DC Bus)"
    )

    print("=" * 60)
    print("Aircraft Electrical System Health Monitor")
    print("=" * 60)
    print(f"Bus Mode:          {bus_mode}")
    print(f"Nominal Voltage:   {nom_v}")
    print(f"Nominal Frequency: {nom_f}")
    print(f"Duration:          {duration:.1f} s @ {config.sampling_rate_hz:.1f} Hz")
    print(f"Random Seed:       {seed}")
    print("-" * 60)

    simulator = ElectricalBusSimulator(config=config, logger=logger)
    injector = FaultInjector(config=config, logger=logger)
    preprocessor = DataPreprocessor(config=config, logger=logger)
    threshold_monitor = EngineeringThresholdMonitor(config=config, logger=logger)
    ml_detector = IsolationForestDetector(config=config, logger=logger)
    health_monitor = IntegratedHealthMonitor(config=config, logger=logger)
    evaluator = ModelEvaluator(logger=logger)
    reporter = ReportGenerator(config=config, logger=logger)
    visualizer = Visualizer(config=config, logger=logger)

    # 3. Generate Normal Baseline Dataset for Model Training (Unsupervised)
    print("Generating baseline normal operating data for model fitting...")
    df_normal_train = simulator.generate(
        duration_seconds=max(300.0, duration * 0.5),
        sampling_rate_hz=config.sampling_rate_hz,
        bus_type=config.bus_type,
        seed=seed,
    )
    _, X_train_scaled = preprocessor.process(df_normal_train, is_training=True)

    print("Training Isolation Forest on normal baseline envelope...")
    ml_detector.fit(X_train_scaled)

    # 4. Generate Test Dataset with Injected Controlled Electrical Faults
    print("\nGenerating synthetic operational dataset...")
    df_test_raw = simulator.generate(
        duration_seconds=duration,
        sampling_rate_hz=config.sampling_rate_hz,
        bus_type=config.bus_type,
        seed=seed + 1,
    )
    n_samples = len(df_test_raw)
    print(f"Samples generated: {n_samples}")

    print("Injecting controlled electrical & sensor faults...")
    df_test_injected, events = injector.inject_faults(df_test_raw, seed=seed)
    print(f"Fault events injected: {len(events)}")

    # Save raw dataset
    raw_path = Path("data/raw") / f"sensor_telemetry_{config.bus_type}.csv"
    df_test_injected.to_csv(raw_path, index=False)

    # 5. Execute Pipeline Stages
    print("Running preprocessing and feature engineering...")
    df_processed, X_test_scaled = preprocessor.process(df_test_injected, is_training=False)

    print("Running threshold monitor...")
    df_rules = threshold_monitor.monitor(df_processed)

    print("Running Isolation Forest inference...")
    df_ml = ml_detector.predict(X_test_scaled)

    print("Evaluating integrated health status & health scores...")
    df_master, alerts = health_monitor.evaluate_health(df_processed, df_rules, df_ml)

    # Save processed master dataset
    proc_path = Path("data/processed") / f"health_monitored_{config.bus_type}.csv"
    df_master.to_csv(proc_path, index=False)

    # 6. Quantitative Performance Evaluation
    metrics = evaluator.evaluate_predictions(
        df_master["true_fault"], df_master["final_anomaly"], "Combined Health Monitor"
    )
    comparison_df = evaluator.compare_methods(df_master)
    fault_df = evaluator.evaluate_by_fault_type(df_master)
    event_summary = evaluator.evaluate_events(df_master, events)

    # 7. Generate Reports and Visualizations
    reporter.save_alerts(alerts)
    reporter.save_metrics(comparison_df, fault_df, event_summary)
    summary_report_file = reporter.generate_summary_report(
        df_master, metrics, comparison_df, fault_df, event_summary, events
    )

    print("Generating diagnostic figures and summary dashboard...")
    visualizer.plot_all(df_master, events, metrics, fault_df, event_summary)

    # 8. Print Clean Terminal Summary (Section 42 format)
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:        {metrics.accuracy * 100:.2f}%")
    print(f"Precision:       {metrics.precision * 100:.2f}%")
    print(f"Recall:          {metrics.recall * 100:.2f}%")
    print(f"F1 Score:        {metrics.f1 * 100:.2f}%")
    print(f"Event Detection: {event_summary.event_detection_rate:.1f}%")
    print(f"Avg Detect Delay:{event_summary.mean_detection_delay_sec:.3f} s")

    print("\nFault Detection Breakdown:")
    for _, row in fault_df.iterrows():
        ft = row["fault_type"]
        if "normal" not in ft:
            print(f"  {ft.replace('_', ' ').title():<22}: {row['detection_rate']:>5.1f}%")

    print("\nReports saved to:")
    print("  outputs/reports/")
    print("  outputs/metrics/")
    print("\nFigures saved to:")
    print("  outputs/figures/")
    print("\nPipeline completed successfully.")
    print("=" * 60)

    # Optional Streaming Demonstration (Section 56)
    if is_stream_demo:
        print("\n--- Starting Real-Time Window Streaming Demonstration ---")
        print(f"{'Time [s]':>8} | {'Voltage':>8} | {'Current':>8} | {'Status':>12} | {'Score':>6} | {'Alert'}")
        print("-" * 65)
        # Stream the last 25 samples
        stream_subset = df_master.iloc[100:130]
        for _, row in stream_subset.iterrows():
            t = row["timestamp"]
            v = row["voltage"]
            c = row["current"]
            status = row["health_status"]
            score = row["health_score"]
            diag = row["diagnosed_fault_type"]
            v_str = f"{v:.1f}V" if pd.notna(v) else "NaN"
            print(f"{t:>8.1f} | {v_str:>8} | {c:>7.1f}A | {status:>12} | {score:>6.1f} | {diag}")
            time.sleep(0.04)  # Visual cadence for terminal
        print("--- Real-Time Streaming Demonstration Complete ---\n")


def main() -> None:
    """Entry point parsing CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Aircraft Electrical System Health Monitor - Simulation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["ac", "dc"],
        default="ac",
        help="Electrical bus simulation mode ('ac' = 115V 400Hz, 'dc' = 28V)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Simulation duration in seconds (e.g. 600.0 for 10 minutes)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Run optional real-time streaming demonstration in terminal",
    )

    args = parser.parse_args()
    run_pipeline(
        mode=args.mode,
        duration=args.duration,
        seed=args.seed,
        config_path=args.config,
        is_stream_demo=args.stream,
    )


if __name__ == "__main__":
    main()
