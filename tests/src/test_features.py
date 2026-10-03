"""Unit tests for the src.features module."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.data_prep import coerce_types
from src.features import (
    filter_by_window,
    filter_cancellations,
    build_group_identifiers,
    build_rfm_features,
    build_tenure_feature,
    build_interval_features,
    build_activity_features,
    build_spend_feature,
    build_orders_feature,
    build_distinct_products_feature,
    build_mean_order_quantity_feature,
    build_return_rate_feature,
    build_country_feature,
)

TEST_INTERIM_FRAME = pd.DataFrame(
    {
        "invoice": [
            "123456",
            "345678",
            "456789",
            "456123",
            "567234",
            "678345",
            "987654",
            "876543",
            "765432",
            "C234567",
            "654321",
            "321654",
            "432765",
            "543876",
            "654987",
        ],
        "stock_code": [
            "12345A",
            "12345A",
            "12345A",
            "12345A",
            "23456",
            "23456B",
            "34567",
            "12345A",
            "12345A",
            "23456",
            "45678",
            "23456B",
            "23456",
            "34567A",
            "34567A",
        ],
        "description": ["FILLER"] * 15,
        "quantity": [10, 40, 40, 10, 20, 40, 20, 10, 10, 20, 20, 40, 20, 10, 40],
        "invoice_date": [
            "2025-12-01",
            "2025-12-24",
            "2026-01-03",
            "2026-01-05",
            "2026-01-09",
            "2026-01-12",
            "2026-01-15",
            "2026-01-20",
            "2026-02-02",
            "2026-02-04",
            "2026-02-06",
            "2026-02-13",
            "2026-02-14",
            "2026-02-15",
            "2026-02-21",
        ],
        "price": ([10] * 9) + [-10] + ([10] * 5),
        "customer_id": [
            12345,
            34567,
            34567,
            12345,
            23456,
            34567,
            23456,
            12345,
            12345,
            23456,
            23456,
            34567,
            23456,
            12345,
            34567,
        ],
        "country": ["Germany"] + (["United Kingdom"] * 14),
        "cancelled": ([0] * 9) + [1] + ([0] * 5),
    }
)

TEST_INTERIM_FRAME = coerce_types(TEST_INTERIM_FRAME)
TEST_INTERIM_FRAME["line_total"] = (
    TEST_INTERIM_FRAME["price"] * TEST_INTERIM_FRAME["quantity"]
)

TEST_PROCESSED_FRAME = pd.DataFrame(
    {
        "customer_id": [12345, 23456, 34567],
        "cutoff": ["2026-02-14", "2026-02-14", "2026-02-14"],
        "churn_horizon": [14, 14, 14],
    }
)

TEST_PROCESSED_FRAME["cutoff"] = pd.to_datetime(TEST_PROCESSED_FRAME["cutoff"]).astype(
    "datetime64[us]"
)

TEST_CUTOFF = "2026-02-14"
TEST_ELIGIBILITY_LOOKBACK = 14
TEST_CHURN_HORIZON = 14


def test_filter_by_window():
    test_frame = pd.DataFrame(
        {
            "invoice_date": [
                "2026-01-15",
                "2026-01-20",
                "2026-02-02",
                "2026-02-06",
                "2026-02-13",
                "2026-02-14",
                "2026-02-15",
                "2026-02-21",
                "2026-04-03",
            ]
        }
    )
    test_frame["invoice_date"] = pd.to_datetime(test_frame["invoice_date"]).astype(
        "datetime64[us]"
    )

    expected1 = pd.DataFrame(
        {
            "invoice_date": [
                "2026-02-02",
                "2026-02-06",
                "2026-02-13",
                "2026-02-14",
            ]
        }
    )
    expected1["invoice_date"] = pd.to_datetime(expected1["invoice_date"]).astype(
        "datetime64[us]"
    )
    expected2 = pd.DataFrame(
        {
            "invoice_date": [
                "2026-01-15",
                "2026-01-20",
                "2026-02-02",
                "2026-02-06",
                "2026-02-13",
                "2026-02-14",
            ]
        }
    )
    expected2["invoice_date"] = pd.to_datetime(expected2["invoice_date"]).astype(
        "datetime64[us]"
    )

    test1 = filter_by_window(
        df=test_frame,
        cutoff=TEST_CUTOFF,
        eligibility_lookback=TEST_ELIGIBILITY_LOOKBACK,
    )
    test2 = filter_by_window(
        df=test_frame, cutoff=TEST_CUTOFF, eligibility_lookback=None
    )

    for test, expected in zip([test1, test2], [expected1, expected2]):
        assert test["invoice_date"].max() <= pd.Timestamp(TEST_CUTOFF)
        pdt.assert_series_equal(
            test["invoice_date"].reset_index(drop=True),
            expected["invoice_date"],
            check_dtype=False,
        )


def test_filter_cancellations():
    test_frame = pd.DataFrame(
        {
            "description": ["item_1", "item_2", "item_3", "item_4"],
            "cancelled": [0, 1, 0, 1],
        }
    )
    expected = pd.DataFrame(
        {
            "description": ["item_1", "item_3"],
            "cancelled": [0, 0],
        }
    )
    test = filter_cancellations(df=test_frame)
    for column in ["description", "cancelled"]:
        pdt.assert_series_equal(
            test[column].reset_index(drop=True), expected[column], check_dtype=False
        )


def test_build_group_identifiers():
    test = build_group_identifiers(
        df=TEST_INTERIM_FRAME,
        cutoff=TEST_CUTOFF,
        eligibility_lookback=TEST_ELIGIBILITY_LOOKBACK,
        churn_horizon=TEST_CHURN_HORIZON,
    )
    for column in ["customer_id", "cutoff", "churn_horizon"]:
        pdt.assert_series_equal(
            test[column], TEST_PROCESSED_FRAME[column], check_dtype=False
        )


def test_build_rfm_features():
    test = build_rfm_features(
        df_processed=TEST_PROCESSED_FRAME,
        df_interim=TEST_INTERIM_FRAME,
        cutoff=TEST_CUTOFF,
    )
    expected = pd.DataFrame(
        {
            "recency": [12, 0, 1],
            "frequency": [4, 4, 4],
            "monetary": [400, 800, 1600],
            "mean_order_value": [100, 200, 400],
        }
    )
    for column in ["recency", "frequency", "monetary", "mean_order_value"]:
        pdt.assert_series_equal(test[column], expected[column], check_dtype=False)


def test_build_tenure_feature():
    test = build_tenure_feature(
        df_processed=TEST_PROCESSED_FRAME,
        df_interim=TEST_INTERIM_FRAME,
        cutoff=TEST_CUTOFF,
    )
    expected = pd.DataFrame({"tenure": [75, 36, 52]})
    pdt.assert_series_equal(test["tenure"], expected["tenure"], check_dtype=False)


def test_build_interval_features():
    test_frame = TEST_PROCESSED_FRAME.copy()
    test_frame["recency"] = np.array([12, 0, 1])
    test = build_interval_features(
        df_processed=test_frame, df_interim=TEST_INTERIM_FRAME, cutoff=TEST_CUTOFF
    )
    expected = pd.DataFrame(
        {
            "avg_inter_purchase_gap": [21, 12, 17],
            "std_inter_purchase_gap": [12.17, 8.72, 13.00],
            "recency_ratio": [0.57, 0, 0.06],
        }
    )
    for column in ["avg_inter_purchase_gap", "std_inter_purchase_gap", "recency_ratio"]:
        test_values = test[column].to_numpy()
        expected_values = expected[column].to_numpy()
        assert test_values == pytest.approx(expected_values, abs=0.01)


def test_build_spend_feature():
    df_filtered = filter_by_window(TEST_INTERIM_FRAME, TEST_CUTOFF, 7)
    df_filtered = filter_cancellations(df_filtered)
    test = build_spend_feature(df=df_filtered, days=7)
    expected = pd.Series([200, 400], name="spend_7d")
    pdt.assert_series_equal(test.reset_index(drop=True), expected, check_dtype=False)


def test_build_orders_feature():
    df_filtered = filter_by_window(TEST_INTERIM_FRAME, TEST_CUTOFF, 7)
    df_filtered = filter_cancellations(df_filtered)
    test = build_orders_feature(df=df_filtered, days=7)
    expected = pd.Series([1, 1], name="orders_7d")
    pdt.assert_series_equal(test.reset_index(drop=True), expected, check_dtype=False)


def test_build_activity_features():
    test_processed_frame = pd.DataFrame(
        {
            "customer_id": [12345],
            "cutoff": ["2026-02-14"],
            "churn_horizon": [14],
        }
    )

    test_interim_frame = pd.DataFrame(
        {
            "invoice": [
                "123456",
                "234567",
                "345678",
                "456789",
                "987654",
                "876543",
                "765432",
                "654321",
            ],
            "stock_code": [
                "12345A",
                "23456B",
                "34567C",
                "45678D",
                "56789E",
                "98765F",
                "87654G",
                "76543H",
            ],
            "description": ["FILLER"] * 8,
            "quantity": [10] * 8,
            "invoice_date": [
                "2026-02-7",
                "2026-02-8",
                "2026-02-9",
                "2026-02-10",
                "2026-02-11",
                "2026-02-12",
                "2026-02-13",
                "2026-02-14",
            ],
            "price": [10] * 8,
            "customer_id": [12345] * 8,
            "country": ["United Kingdom"] * 8,
            "cancelled": [0] * 8,
        }
    )

    test_interim_frame = coerce_types(test_interim_frame)
    test_interim_frame["line_total"] = (
        test_interim_frame["price"] * test_interim_frame["quantity"]
    )

    test = build_activity_features(
        df_processed=test_processed_frame,
        df_interim=test_interim_frame,
        cutoff="2026-02-14",
        lookback_days=[2, 4, 8],
    )

    assert set(test.columns) == {
        "customer_id",
        "cutoff",
        "churn_horizon",
        "spend_2d",
        "spend_4d",
        "spend_8d",
        "orders_2d",
        "orders_4d",
        "orders_8d",
        "spend_momentum_2d_4d",
        "spend_momentum_4d_8d",
        "order_momentum_2d_4d",
        "order_momentum_4d_8d",
    }
    assert int(test.loc[0, "spend_momentum_2d_4d"]) == 1
    assert int(test.loc[0, "spend_momentum_4d_8d"]) == 1
    assert int(test.loc[0, "order_momentum_2d_4d"]) == 1
    assert int(test.loc[0, "order_momentum_4d_8d"]) == 1


def test_build_distinct_products_feature():
    test = build_distinct_products_feature(
        df_processed=TEST_PROCESSED_FRAME,
        df_interim=TEST_INTERIM_FRAME,
        cutoff=TEST_CUTOFF,
    )
    expected = pd.DataFrame({"distinct_products": [1, 3, 2]})
    pdt.assert_series_equal(
        test["distinct_products"], expected["distinct_products"], check_dtype=False
    )


def test_build_mean_order_quantity_feature():
    test = build_mean_order_quantity_feature(
        df_processed=TEST_PROCESSED_FRAME,
        df_interim=TEST_INTERIM_FRAME,
        cutoff=TEST_CUTOFF,
    )
    expected = pd.DataFrame({"mean_order_quantity": [10, 20, 40]})
    pdt.assert_series_equal(
        test["mean_order_quantity"], expected["mean_order_quantity"], check_dtype=False
    )


def test_build_return_rate_feature():
    test = build_return_rate_feature(
        df_processed=TEST_PROCESSED_FRAME,
        df_interim=TEST_INTERIM_FRAME,
        cutoff=TEST_CUTOFF,
    )
    expected_values = np.array([0, 0.2, 0])
    test_values = test["return_rate"].to_numpy()
    assert test_values == pytest.approx(expected_values)


def test_build_country_feature():
    test = build_country_feature(
        df_processed=TEST_PROCESSED_FRAME,
        df_interim=TEST_INTERIM_FRAME,
        cutoff=TEST_CUTOFF,
    )
    expected = pd.DataFrame({"country": ["United Kingdom"] * 3})
    pdt.assert_series_equal(test["country"], expected["country"], check_dtype=False)
