"""
Performance Evaluation and Benchmark Validation Engine.

Evaluates anomaly detection efficacy against ground-truth labels at three distinct scopes:
  1. Sample-level metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, FPR
  2. Fault-level metrics: Breakdown of detection rates per individual electrical fault type
  3. Event-level metrics: Event detection success rate and time-to-first-detect latency
  4. System benchmark comparison: Threshold-only vs Isolation Forest vs Combined System

STRICT HONESTY REQUIREMENT:
All metrics are mathematically derived from actual experiment executions. Zero fabricated numbers.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.fault_injection import FaultEvent


@dataclass
class ClassificationMetrics:
    """Standard binary classification evaluation metrics."""

    method_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    false_positive_rate: float
    total_samples: int
    fault_samples: int
    normal_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventEvaluationSummary:
    """Summary of event-level detection efficacy and detection latency."""

    total_events: int
    detected_events: int
    missed_events: int
    event_detection_rate: float
    mean_detection_delay_sec: float
    event_details: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "detected_events": self.detected_events,
            "missed_events": self.missed_events,
            "event_detection_rate": self.event_detection_rate,
            "mean_detection_delay_sec": self.mean_detection_delay_sec,
            "event_details": self.event_details,
        }


class ModelEvaluator:
    """Evaluates sample-level, fault-level, and event-level detection performance."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def evaluate_predictions(
        self, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, method_name: str = "Combined System"
    ) -> ClassificationMetrics:
        """Compute standard classification metrics from ground truth and predictions."""
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)

        acc = float(accuracy_score(y_t, y_p))
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))

        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return ClassificationMetrics(
            method_name=method_name,
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            false_positive_rate=round(fpr, 4),
            total_samples=len(y_t),
            fault_samples=int(np.sum(y_t == 1)),
            normal_samples=int(np.sum(y_t == 0)),
        )

    def compare_methods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Benchmark comparison: Thresholds vs Isolation Forest vs Combined System."""
        y_true = df["true_fault"].values

        m_rule = self.evaluate_predictions(y_true, df["rule_anomaly"].values, "Threshold Rules")
        m_ml = self.evaluate_predictions(y_true, df["ml_anomaly"].values, "Isolation Forest")
        m_comb = self.evaluate_predictions(y_true, df["final_anomaly"].values, "Combined Health Monitor")

        rows = [m_rule.to_dict(), m_ml.to_dict(), m_comb.to_dict()]
        comparison_df = pd.DataFrame(rows)
        return comparison_df

    def evaluate_by_fault_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Break down detection rate across each distinct ground-truth fault type."""
        records: List[Dict[str, Any]] = []

        # Only evaluate on fault observations
        fault_df = df[df["true_fault"] == 1]
        fault_types = sorted(fault_df["fault_type"].unique())

        for ft in fault_types:
            subset = fault_df[fault_df["fault_type"] == ft]
            n_samples = len(subset)
            detected = int(np.sum(subset["final_anomaly"] == 1))
            missed = n_samples - detected
            det_rate = round((detected / n_samples) * 100.0, 2) if n_samples > 0 else 0.0

            records.append(
                {
                    "fault_type": ft,
                    "samples": n_samples,
                    "detected": detected,
                    "missed": missed,
                    "detection_rate": det_rate,
                }
            )

        # Include normal baseline performance
        norm_subset = df[df["true_fault"] == 0]
        n_norm = len(norm_subset)
        fp_samples = int(np.sum(norm_subset["final_anomaly"] == 1))
        tn_samples = n_norm - fp_samples
        records.append(
            {
                "fault_type": "normal (false alarms)",
                "samples": n_norm,
                "detected": fp_samples,  # in this row, 'detected' means false positive
                "missed": tn_samples,    # correctly normal
                "detection_rate": round((fp_samples / n_norm) * 100.0, 2) if n_norm > 0 else 0.0,
            }
        )

        return pd.DataFrame(records)

    def evaluate_events(
        self, df: pd.DataFrame, events: List[FaultEvent]
    ) -> EventEvaluationSummary:
        """Evaluate event-level detection efficacy and detection latency."""
        if not events:
            return EventEvaluationSummary(0, 0, 0, 0.0, 0.0, [])

        event_details: List[Dict[str, Any]] = []
        delays: List[float] = []
        detected_count = 0

        for ev in events:
            ev_df = df.iloc[ev.start_idx : ev.end_idx]
            detected_indices = ev_df.index[ev_df["final_anomaly"] == 1].tolist()

            is_detected = len(detected_indices) > 0
            if is_detected:
                detected_count += 1
                first_idx = detected_indices[0]
                first_time = float(df.loc[first_idx, "timestamp"])
                delay = max(0.0, first_time - ev.start_time)
                delays.append(delay)
            else:
                delay = np.nan

            event_details.append(
                {
                    "fault_id": ev.fault_id,
                    "fault_type": ev.fault_type,
                    "severity": ev.severity,
                    "start_time": round(ev.start_time, 2),
                    "end_time": round(ev.end_time, 2),
                    "duration_sec": round(ev.end_time - ev.start_time, 2),
                    "detected": is_detected,
                    "detection_delay_sec": round(delay, 3) if not np.isnan(delay) else None,
                }
            )

        total_ev = len(events)
        rate = round((detected_count / total_ev) * 100.0, 2)
        mean_delay = round(float(np.mean(delays)), 3) if delays else 0.0

        return EventEvaluationSummary(
            total_events=total_ev,
            detected_events=detected_count,
            missed_events=total_ev - detected_count,
            event_detection_rate=rate,
            mean_detection_delay_sec=mean_delay,
            event_details=event_details,
        )
