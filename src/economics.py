"""Selects a contact threshold that maximises the expected value of a retention campaign based on the campaign capacity."""

from typing import Any

import numpy as np
import pandas as pd

from src.config import EconomicsCfg

THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)


def scale_capacity(capacity: int, n_scored: int, n_evaluated: int) -> int:
    """Scales campaign capacity (determined based on n_scored) into the equivalent count on the evaluation dataset (n_evaluated)."""
    if n_scored <= 0:
        return min(capacity, n_evaluated)
    return max(1, min(int(round((capacity / n_scored) * n_evaluated)), n_evaluated))


def confusion_counts(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, int]:
    """Counts the number of true positives, false positives, true negatives and false negatives at a single threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    flagged_positives = y_proba >= threshold
    flagged_negatives = y_proba < threshold
    return {
        "tp": int((flagged_positives & (y_true == 1)).sum()),
        "fp": int((flagged_positives & (y_true == 0)).sum()),
        "tn": int((flagged_negatives & (y_true == 0)).sum()),
        "fn": int((flagged_negatives & (y_true == 1)).sum()),
    }


def calculate_expected_value(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    save_rate: float,
    customer_lifetime_value: float,
    cost: float,
) -> float:
    """Calculates total expected campaign value at a single threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    counts = confusion_counts(y_true, y_proba, threshold)
    n_flagged = counts["tp"] + counts["fp"]
    return (counts["tp"] * save_rate * customer_lifetime_value) - (n_flagged * cost)


def top_n_campaign_value(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n: int,
    save_rate: float,
    customer_lifetime_value: float,
    cost: float,
) -> dict:
    """Calculates the value of contacting the top n customers by score, regardless of profitability."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    k = min(int(n), y_true.size)
    if k == 0:
        return {
            "n_flagged": 0,
            "chosen_threshold": 1.0,
            "tp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "expected_value": 0.0,
        }

    top_k = np.argsort(y_proba, kind="stable")[-k:]
    chosen_threshold = float(y_proba[top_k].min())

    counts = confusion_counts(y_true, y_proba, chosen_threshold)
    n_flagged = counts["tp"] + counts["fp"]
    total_churns = int(y_true.sum())

    return {
        "n_flagged": n_flagged,
        "chosen_threshold": chosen_threshold,
        "tp": counts["tp"],
        "precision": counts["tp"] / n_flagged,
        "recall": counts["tp"] / total_churns if total_churns else 0.0,
        "expected_value": calculate_expected_value(
            y_true=y_true,
            y_proba=y_proba,
            threshold=chosen_threshold,
            save_rate=save_rate,
            customer_lifetime_value=customer_lifetime_value,
            cost=cost,
        ),
    }


def generate_expected_value_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    save_rate: float,
    customer_lifetime_value: float,
    cost: float,
) -> list[dict[str, Any]]:
    """Generates an expected value curve over a range of thresholds."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    rows = []
    total_churns = int(y_true.sum())
    for threshold in thresholds:
        counts = confusion_counts(y_true, y_proba, threshold)
        n_flagged = counts["tp"] + counts["fp"]
        if n_flagged == 0:
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "n_flagged": n_flagged,
                "tp": counts["tp"],
                "precision": counts["tp"] / n_flagged,
                "recall": counts["tp"] / total_churns if total_churns else 0.0,
                "expected_value": calculate_expected_value(
                    y_true=y_true,
                    y_proba=y_proba,
                    threshold=threshold,
                    save_rate=save_rate,
                    customer_lifetime_value=customer_lifetime_value,
                    cost=cost,
                ),
            }
        )
    return rows


def select_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    save_rate: float,
    customer_lifetime_value: float,
    cost: float,
    raw_capacity: int,
    scaled_capacity: int,
) -> dict[str, Any]:
    """Selects an optimal threshold from the expected value curve based on campaign capacity."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    curve = generate_expected_value_curve(
        y_true=y_true,
        y_proba=y_proba,
        thresholds=thresholds,
        save_rate=save_rate,
        customer_lifetime_value=customer_lifetime_value,
        cost=cost,
    )
    curve = pd.DataFrame(curve)
    chosen_threshold_row = curve.loc[curve["expected_value"].idxmax()]
    capacity_binding = bool(chosen_threshold_row["n_flagged"] > scaled_capacity)

    if not capacity_binding:
        return {
            "mode": "expected_value",
            "capacity_binding": False,
            "chosen_threshold": float(chosen_threshold_row["threshold"]),
            "n_flagged": int(chosen_threshold_row["n_flagged"]),
            "expected_value": float(chosen_threshold_row["expected_value"]),
            "unconstrained_n_flagged": int(chosen_threshold_row["n_flagged"]),
            "unconstrained_expected_value": float(
                chosen_threshold_row["expected_value"]
            ),
            "unconstrained_threshold": float(chosen_threshold_row["threshold"]),
            "precision": float(chosen_threshold_row["precision"]),
            "recall": float(chosen_threshold_row["recall"]),
            "raw_capacity": int(raw_capacity),
            "scaled_capacity": int(scaled_capacity),
            "n_evaluated": int(y_true.size),
            "forgone_value": 0.0,
        }

    top_n_campaign = top_n_campaign_value(
        y_true=y_true,
        y_proba=y_proba,
        n=scaled_capacity,
        save_rate=save_rate,
        customer_lifetime_value=customer_lifetime_value,
        cost=cost,
    )

    return {
        "mode": "capacity",
        "capacity_binding": True,
        "chosen_threshold": top_n_campaign["chosen_threshold"],
        "n_flagged": top_n_campaign["n_flagged"],
        "expected_value": top_n_campaign["expected_value"],
        "unconstrained_n_flagged": int(chosen_threshold_row["n_flagged"]),
        "unconstrained_expected_value": float(chosen_threshold_row["expected_value"]),
        "unconstrained_threshold": float(chosen_threshold_row["threshold"]),
        "precision": top_n_campaign["precision"],
        "recall": top_n_campaign["recall"],
        "raw_capacity": int(raw_capacity),
        "scaled_capacity": int(scaled_capacity),
        "n_evaluated": int(y_true.size),
        "forgone_value": float(chosen_threshold_row["expected_value"])
        - top_n_campaign["expected_value"],
    }


def build_sensitivity_grid(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    save_rate: float,
    customer_lifetime_value: float,
    cost: float,
    sensitivity_spread: float,
    sensitivity_points: int,
) -> dict[str, float]:
    """Calculates range of expected campaign values for sensitivity (as save_rate is an estimate not informed by the dataset)."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    lower_bound = max(0.0, (save_rate - sensitivity_spread))
    upper_bound = min(max(0.0, (save_rate + sensitivity_spread)), 1.0)
    rates = np.linspace(lower_bound, upper_bound, sensitivity_points)

    return {
        f"{rate}": calculate_expected_value(
            y_true=y_true,
            y_proba=y_proba,
            threshold=threshold,
            save_rate=rate,
            customer_lifetime_value=customer_lifetime_value,
            cost=cost,
        )
        for rate in rates
    }


def calculate_break_even_save_rate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    customer_lifetime_value: float,
    cost: float,
) -> float | None:
    """Calculates the break-even save rate for the campaign."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    counts = confusion_counts(y_true, y_proba, threshold)
    n_flagged = counts["tp"] + counts["fp"]
    return (
        (n_flagged * cost) / (counts["tp"] * customer_lifetime_value)
        if counts["tp"]
        else None
    )


def build_campaign_summary(
    save_rate: float,
    customer_lifetime_value: float,
    cost: float,
    capacity: int,
    threshold: dict[str, Any],
    sensitivity_grid: dict[str, float],
    break_even_save_rate: float | None,
) -> dict:
    """Builds a dictionary summarising the campaign performance"""
    return {
        "assumptions": {
            "cost_per_contact": cost,
            "average_customer_lifetime_value": customer_lifetime_value,
            "save_rate": save_rate,
            "campaign_capacity": capacity,
        },
        **threshold,
        "expected_value_by_save_rate": sensitivity_grid,
        "break_even_save_rate": break_even_save_rate,
        "break_even_probability": cost / (save_rate * customer_lifetime_value),
    }


def select_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_scored: int,
    cost: float,
    customer_lifetime_value: float,
    save_rate: float,
    sensitivity_spread: float,
    sensitivity_points: int,
    capacity: int,
) -> dict[str, Any]:
    """Engine for selecting a contact threshold by expected value, subject to capacity."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    scaled_capacity = scale_capacity(
        capacity=capacity, n_scored=n_scored, n_evaluated=y_true.size
    )

    chosen_threshold = select_optimal_threshold(
        y_true=y_true,
        y_proba=y_proba,
        thresholds=THRESHOLD_GRID,
        save_rate=save_rate,
        customer_lifetime_value=customer_lifetime_value,
        cost=cost,
        raw_capacity=capacity,
        scaled_capacity=scaled_capacity,
    )

    sensitivity_grid = build_sensitivity_grid(
        y_true=y_true,
        y_proba=y_proba,
        threshold=chosen_threshold["chosen_threshold"],
        save_rate=save_rate,
        sensitivity_spread=sensitivity_spread,
        sensitivity_points=sensitivity_points,
        customer_lifetime_value=customer_lifetime_value,
        cost=cost,
    )

    break_even_save_rate = calculate_break_even_save_rate(
        y_true=y_true,
        y_proba=y_proba,
        threshold=chosen_threshold["chosen_threshold"],
        customer_lifetime_value=customer_lifetime_value,
        cost=cost,
    )

    return build_campaign_summary(
        save_rate=save_rate,
        customer_lifetime_value=customer_lifetime_value,
        cost=cost,
        capacity=capacity,
        threshold=chosen_threshold,
        sensitivity_grid=sensitivity_grid,
        break_even_save_rate=break_even_save_rate,
    )
