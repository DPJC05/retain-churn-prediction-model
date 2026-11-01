"""Generates curve data for the Chart.js frontend using y_true and y_proba."""

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def roc_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    """Generates receiver operating characteristics (ROC) curve and baseline data for Chart.js."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape != y_proba.shape:
        raise ValueError("shape mismatch between y_true and y_proba")

    if int(y_true.sum()) in (0, int(y_true.size)):
        return {
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "baseline": [],
            "roc_auc": None,
            "n_rows": int(y_true.size),
        }

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    # roc_curve() sets the first threshold to np.inf to anchor the ROC curve at the origin (0, 0).
    # This step keeps the origin anchor, but replaces np.inf (which is not valid JSON) with 1.0
    # (functionally equivalent, as it is the fpr and tpr upper boundary).
    thresholds = thresholds.copy()
    thresholds[0] = 1.0

    return {
        "fpr": fpr.tolist(),  # x-axis values
        "tpr": tpr.tolist(),  # y-axis values
        "thresholds": thresholds.tolist(),
        # The random baseline for the ROC curve is always 0.50.
        # This translates to the straight line from (0, 0) to (1, 1): y = x (tpr = fpr).
        # As such, the baseline curve data is equivalent to the x-axis values (fpr).
        "baseline": fpr.tolist(),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "n_rows": int(y_true.size),
    }


def pr_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    """Generates precision-recall (PR) curve and baseline data for Chart.js."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape != y_proba.shape:
        raise ValueError("shape mismatch between y_true and y_proba")

    base_rate = float(y_true.mean())

    if int(y_true.sum()) in (0, int(y_true.size)):
        return {
            "precision": [],
            "recall": [],
            "thresholds": [],
            "baseline": [],
            "base_rate": base_rate,
            "pr_auc": None,
            "n_rows": int(y_true.size),
        }

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # precision_recall_curve() returns one more precision (1.0) point and recall point (0.0) as compared to thresholds.
    # i.e., the relative np.ndarray shapes are: (precision.shape == recall.shape) AND (precision.shape > threshold.shape).
    # This step drops the extra points to ensure all the np.ndarrays have the same shape.
    precision, recall = precision[:-1], recall[:-1]

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        # Baseline for PR-AUC is a horizontal line at the average churn rate of the evaluated dataset.
        "baseline": [base_rate] * len(recall),
        "base_rate": base_rate,
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "n_rows": int(y_true.size),
    }


def gains_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    """Generates cumulative gains curve and baseline data for Chart.js."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape != y_proba.shape:
        raise ValueError("shape mismatch between y_true and y_proba")

    n_rows = int(y_true.size)
    total_churns = int(y_true.sum())

    if (total_churns == 0) or (n_rows == 0):
        return {
            "contacted": [],
            "captured": [],
            "baseline": [],
            "n_rows": n_rows,
        }

    order = np.argsort(-y_proba, kind="stable")
    captured = np.cumsum(y_true[order]) / total_churns
    contacted = np.arange(1, n_rows + 1) / n_rows

    return {
        "contacted": contacted.tolist(),
        "captured": captured.tolist(),
        "baseline": contacted.tolist(),  # Same baseline logic as ROC curve
        "n_rows": n_rows,
    }


def calibration_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict[str, Any]:
    """Generates calibration curve and baseline data."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape != y_proba.shape:
        raise ValueError("shape mismatch between y_true and y_proba")

    if int(y_true.sum()) in (0, int(y_true.size)):
        return {
            "mean_predicted": [],
            "observed_frequency": [],
            "bin_counts": [],
            "baseline": [],
            "n_rows": int(y_true.size),
        }

    observed_frequency, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy=strategy
    )

    edges = (mean_predicted[:-1] + mean_predicted[1:]) / 2
    bin_index = np.searchsorted(edges, y_proba)
    bin_counts = np.bincount(bin_index, minlength=mean_predicted.size)

    return {
        "mean_predicted": mean_predicted.tolist(),
        "observed_frequency": observed_frequency.tolist(),
        "bin_counts": bin_counts.tolist(),
        # The calibration curve baseline represents a perfect model: y = x (observed_frequency = mean_predicted).
        # As such, the baseline curve data is equivalent to the x-axis values (mean_predicted).
        "baseline": mean_predicted.tolist(),
        "n_rows": int(y_true.size),
    }


def lift_by_decile(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Generates per-decile precision, lift and cumulative recall, with the evaluated dataset sorted by
    descending churn risk (highest risk in first position).
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape != y_proba.shape:
        raise ValueError("shape mismatch between y_true and y_proba")

    n_rows = int(y_true.size)
    total_churns = int(y_true.sum())

    if (n_rows < 10) or (total_churns == 0):
        return []

    base_rate = total_churns / n_rows

    # Sorts y_true and y_proba in descending order (highest churn risk in 1st position)
    order = np.argsort(-y_proba, kind="stable")
    sorted_labels = y_true[order]
    sorted_scores = y_proba[order]

    # Splits y_true (labels) and y_proba (scores) into deciles
    label_deciles = np.array_split(sorted_labels, 10)
    score_deciles = np.array_split(sorted_scores, 10)

    rows = []
    cumulative_churns = 0
    for i, (labels, scores) in enumerate(zip(label_deciles, score_deciles), start=1):
        churns = int(labels.sum())
        cumulative_churns += churns
        precision = churns / labels.size
        rows.append(
            {
                "decile": i,
                "n_rows": int(labels.size),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
                "n_churned": churns,
                "precision": precision,
                "lift": precision / base_rate if base_rate else None,
                "cumulative_recall": cumulative_churns / total_churns,
            }
        )
    return rows


def generate_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Engine for generating the curve data as a Python dictionary."""

    roc = roc_curve_data(y_true=y_true, y_proba=y_proba)

    pr = pr_curve_data(y_true=y_true, y_proba=y_proba)

    calibration = calibration_data(y_true=y_true, y_proba=y_proba)

    gains = gains_curve_data(y_true=y_true, y_proba=y_proba)

    lift = lift_by_decile(y_true=y_true, y_proba=y_proba)

    return {
        "roc_curve": roc,
        "pr_curve": pr,
        "calibration": calibration,
        "gains_curve": gains,
        "lift": lift,
    }
