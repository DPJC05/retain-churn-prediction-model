"""Unit tests for the src.labelling module."""

import pandas as pd
import pandas.testing as pdt

from src.labelling import (
    get_eligible_customers,
    filter_by_outcome_window,
    build_churn_label,
)

TEST_INTERIM_FRAME = pd.DataFrame(
    {
        "invoice": [
            "123456",
            "C234567",
            "345678",
            "456789",
            "654321",
            "765432",
            "876543",
            "987654",
        ],
        "invoice_date": [
            "2026-02-02",
            "2026-02-06",
            "2026-02-13",
            "2026-02-15",
            "2026-02-18",
            "2026-02-21",
            "2026-02-23",
            "2026-04-03",
        ],
        "customer_id": [12345, 23456, 45678, 34567, 23456, 45678, 34567, 12345],
        "cancelled": [0, 1, 0, 0, 0, 0, 0, 0],
    }
)

TEST_INTERIM_FRAME["invoice_date"] = pd.to_datetime(
    TEST_INTERIM_FRAME["invoice_date"]
).astype("datetime64[us]")

TEST_PROCESSED_FRAME = pd.DataFrame(
    {"customer_id": [12345, 45678], "cutoff": ["2026-02-14", "2026-02-14"]}
)

TEST_PROCESSED_FRAME["cutoff"] = pd.to_datetime(TEST_PROCESSED_FRAME["cutoff"]).astype(
    "datetime64[us]"
)

TEST_CUTOFF = "2026-02-14"
TEST_ELIGIBILITY_LOOKBACK = 14
TEST_CHURN_HORIZON = 14


def test_get_eligible_customers():
    test = get_eligible_customers(
        df=TEST_INTERIM_FRAME,
        cutoff=TEST_CUTOFF,
        eligibility_lookback=TEST_ELIGIBILITY_LOOKBACK,
    )
    for column in ["customer_id", "cutoff"]:
        pdt.assert_series_equal(
            test[column].reset_index(drop=True),
            TEST_PROCESSED_FRAME[column],
            check_dtype=False,
        )


def test_filter_by_outcome_window():
    test = filter_by_outcome_window(
        df=TEST_INTERIM_FRAME, cutoff=TEST_CUTOFF, churn_horizon=TEST_CHURN_HORIZON
    )
    expected = pd.DataFrame(
        {
            "invoice": ["456789", "654321", "765432", "876543"],
            "invoice_date": ["2026-02-15", "2026-02-18", "2026-02-21", "2026-02-23"],
            "customer_id": [34567, 23456, 45678, 34567],
            "cancelled": [0, 0, 0, 0],
        }
    )
    expected["invoice_date"] = pd.to_datetime(expected["invoice_date"]).astype(
        "datetime64[us]"
    )
    for column in ["invoice", "invoice_date", "customer_id", "cancelled"]:
        pdt.assert_series_equal(
            test[column].reset_index(drop=True), expected[column], check_dtype=False
        )


def test_build_churn_label():
    test = build_churn_label(
        df_labels=TEST_PROCESSED_FRAME,
        df_interim=TEST_INTERIM_FRAME,
        cutoff=TEST_CUTOFF,
        churn_horizon=TEST_CHURN_HORIZON,
    )
    expected = pd.DataFrame(
        {
            "customer_id": [12345, 45678],
            "cutoff": ["2026-02-14", "2026-02-14"],
            "churned": [1, 0],
        }
    )
    pdt.assert_series_equal(test["churned"], expected["churned"], check_dtype=False)
