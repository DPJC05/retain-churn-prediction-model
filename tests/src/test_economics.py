"""Unit tests for the src.economics module."""

import numpy as np
import pytest

from src.economics import (
    scale_capacity,
    confusion_counts,
    calculate_expected_value,
    top_n_campaign_value,
    generate_expected_value_curve,
    select_optimal_threshold,
    calculate_break_even_save_rate,
    THRESHOLD_GRID,
)

Y_TRUE = np.array([1, 0, 1, 0, 1, 0])
Y_PROBA = np.array([0.7, 0.4, 0.2, 0.6, 0.8, 0.1])
THRESHOLD = 0.5
SAVE_RATE = 0.15
CUSTOMER_LIFETIME_VALUE = 20
COST = 0.50


def test_scale_capacity():
    test1 = scale_capacity(capacity=10, n_scored=0, n_evaluated=5)
    test2 = scale_capacity(capacity=500, n_scored=5000, n_evaluated=2000)
    assert test1 == 5
    assert test2 == 200


def test_confusion_counts():
    test = confusion_counts(y_true=Y_TRUE, y_proba=Y_PROBA, threshold=THRESHOLD)
    assert test["tp"] == 2
    assert test["fp"] == 1
    assert test["tn"] == 2
    assert test["fn"] == 1


def test_calculate_expected_value():
    test1 = calculate_expected_value(
        y_true=Y_TRUE,
        y_proba=Y_PROBA,
        threshold=THRESHOLD,
        save_rate=SAVE_RATE,
        customer_lifetime_value=CUSTOMER_LIFETIME_VALUE,
        cost=COST,
    )
    test2 = calculate_expected_value(
        y_true=Y_TRUE,
        y_proba=Y_PROBA,
        threshold=0.0,
        save_rate=SAVE_RATE,
        customer_lifetime_value=CUSTOMER_LIFETIME_VALUE,
        cost=COST,
    )
    test3 = calculate_expected_value(
        y_true=Y_TRUE,
        y_proba=Y_PROBA,
        threshold=1.01,
        save_rate=SAVE_RATE,
        customer_lifetime_value=CUSTOMER_LIFETIME_VALUE,
        cost=COST,
    )
    assert test1 == (2 * SAVE_RATE * CUSTOMER_LIFETIME_VALUE) - (3 * COST)
    assert test2 == (3 * SAVE_RATE * CUSTOMER_LIFETIME_VALUE) - (6 * COST)
    assert test3 == 0


def test_top_n_campaign_value():
    test1 = top_n_campaign_value(
        y_true=Y_TRUE,
        y_proba=Y_PROBA,
        n=0,
        save_rate=SAVE_RATE,
        customer_lifetime_value=CUSTOMER_LIFETIME_VALUE,
        cost=COST,
    )
    expected1 = {
        "n_flagged": 0,
        "chosen_threshold": 1.0,
        "tp": 0,
        "precision": 0.0,
        "recall": 0.0,
        "expected_value": 0.0,
    }

    test2 = top_n_campaign_value(
        y_true=Y_TRUE,
        y_proba=Y_PROBA,
        n=4,
        save_rate=SAVE_RATE,
        customer_lifetime_value=CUSTOMER_LIFETIME_VALUE,
        cost=COST,
    )
    expected2 = {
        "n_flagged": 4,
        "chosen_threshold": 0.4,
        "tp": 2,
        "precision": (2 / 4),
        "recall": (2 / 3),
        "expected_value": (2 * SAVE_RATE * CUSTOMER_LIFETIME_VALUE) - (4 * COST),
    }

    assert test1 == expected1
    assert test2 == pytest.approx(expected2)


def test_generate_expected_value_curve():
    test = generate_expected_value_curve(
        y_true=Y_TRUE,
        y_proba=Y_PROBA,
        thresholds=np.array([0.5]),
        save_rate=SAVE_RATE,
        customer_lifetime_value=CUSTOMER_LIFETIME_VALUE,
        cost=COST,
    )
    expected = [
        {
            "threshold": 0.5,
            "n_flagged": 3,
            "tp": 2,
            "precision": (2 / 3),
            "recall": (2 / 3),
            "expected_value": (2 * SAVE_RATE * CUSTOMER_LIFETIME_VALUE) - (3 * COST),
        }
    ]
    assert test == pytest.approx(expected)


def test_select_optimal_threshold():
    # Test for capacity binding functionality
    test1 = select_optimal_threshold(
        y_true=Y_TRUE,
        y_proba=Y_PROBA,
        thresholds=THRESHOLD_GRID,
        save_rate=SAVE_RATE,
        customer_lifetime_value=CUSTOMER_LIFETIME_VALUE,
        cost=COST,
        raw_capacity=1,
        scaled_capacity=1,
    )
    assert test1["mode"] == "capacity"
    assert test1["capacity_binding"] is True
    assert test1["n_flagged"] <= 1
    assert test1["forgone_value"] >= 0

    # Test for property: as save_rate rises, expected_value should not decrease
    test2 = [
        select_optimal_threshold(
            y_true=Y_TRUE,
            y_proba=Y_PROBA,
            thresholds=THRESHOLD_GRID,
            save_rate=rate,
            customer_lifetime_value=CUSTOMER_LIFETIME_VALUE,
            cost=COST,
            raw_capacity=500,
            scaled_capacity=200,
        )["expected_value"]
        for rate in (0.05, 0.10, 0.20, 0.30)
    ]
    assert test2 == sorted(test2)


def test_calculate_break_even_save_rate():
    test = calculate_break_even_save_rate(
        y_true=Y_TRUE,
        y_proba=Y_PROBA,
        threshold=THRESHOLD,
        customer_lifetime_value=CUSTOMER_LIFETIME_VALUE,
        cost=COST,
    )
    assert test == pytest.approx((3 * COST) / (2 * CUSTOMER_LIFETIME_VALUE))
