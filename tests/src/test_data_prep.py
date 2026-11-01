"""Unit tests for the src.data_prep module."""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.config import load_config
from src.data_prep import (
    standardise_column_headers,
    drop_duplicate_entries,
    get_identified_customers,
    coerce_types,
    flag_cancellations,
    drop_non_products,
    get_line_total,
)

config = load_config()
COLUMN_MAP = config.prep.column_map
NON_PRODUCTS = config.prep.non_products


def test_standardise_column_headers():
    test_frame = pd.DataFrame(
        {
            "Invoice": [],
            "StockCode": [],
            "Description": [],
            "Quantity": [],
            "InvoiceDate": [],
            "Price": [],
            "Customer ID": [],
            "Country": [],
        }
    )
    expected = pd.DataFrame(
        {
            "invoice": [],
            "stock_code": [],
            "description": [],
            "quantity": [],
            "invoice_date": [],
            "price": [],
            "customer_id": [],
            "country": [],
        }
    )
    test = standardise_column_headers(df=test_frame, column_map=COLUMN_MAP)
    assert list(test.columns) == list(expected.columns)


def test_drop_duplicate_entries():
    test_frame = pd.DataFrame({"invoice": [0, 0, 1, 3, 2, 0]})
    expected = pd.DataFrame({"invoice": [0, 1, 3, 2]})
    test, dropped = drop_duplicate_entries(df=test_frame)
    pdt.assert_frame_equal(test.reset_index(drop=True), expected, check_dtype=False)
    assert dropped == 2


def test_drop_non_products():
    test_frame = pd.DataFrame(
        {"stock_code": ["POST", "12345", "12345A", "D", "BANK CHARGES"]}
    )
    expected = pd.DataFrame({"stock_code": ["12345", "12345A"]})
    test, dropped = drop_non_products(df=test_frame, non_products=NON_PRODUCTS)
    pdt.assert_series_equal(
        test["stock_code"].reset_index(drop=True),
        expected["stock_code"],
        check_dtype=False,
    )
    assert dropped == 3


def test_get_identified_customers():
    test_frame = pd.DataFrame({"customer_id": [np.nan, 12345, np.nan, 23456, np.nan]})
    expected = pd.DataFrame({"customer_id": [12345, 23456]})
    test, dropped = get_identified_customers(df=test_frame)
    pdt.assert_frame_equal(test.reset_index(drop=True), expected, check_dtype=False)
    assert dropped == 3


def test_coerce_types():
    test_frame = pd.DataFrame(
        {
            "invoice": [123456, "C234567", "345678"],
            "stock_code": [12345, "POST", "12345A"],
            "description": ["FILLER", "FILLER", 0],
            "quantity": [1, 2, "3"],
            "invoice_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "price": [1, "2", 3],
            "customer_id": [12345, "23456", 34567],
            "country": ["FILLER", "FILLER", 0],
        }
    )
    test = coerce_types(df=test_frame)
    assert test["invoice"].dtype == "string"
    assert test["stock_code"].dtype == "string"
    assert test["description"].dtype == "string"
    assert test["quantity"].dtype == "int32"
    assert test["invoice_date"].dtype == "datetime64[us]"
    assert test["price"].dtype == "float64"
    assert test["customer_id"].dtype == "int64"
    assert test["country"].dtype == "string"


def test_flag_cancellations():
    test_frame = pd.DataFrame({"invoice": ["123456", "234567", "C345678"]})
    expected = pd.DataFrame(
        {"invoice": ["123456", "234567", "C345678"], "cancelled": [0, 0, 1]}
    )
    test = flag_cancellations(df=test_frame)
    pdt.assert_series_equal(test["cancelled"], expected["cancelled"], check_dtype=False)


def test_get_line_total():
    test_frame = pd.DataFrame({"price": [1, 2, 3], "quantity": [4, 5, 6]})
    expected = pd.DataFrame(
        {"price": [1, 2, 3], "quantity": [4, 5, 6], "line_total": [4, 10, 18]}
    )
    test = get_line_total(df=test_frame)
    pdt.assert_series_equal(
        test["line_total"], expected["line_total"], check_dtype=False
    )
