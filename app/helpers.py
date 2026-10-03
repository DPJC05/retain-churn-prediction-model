"""Module containing helper functions for app.routes and app.api."""

import math
from typing import Any

from flask import current_app
import numpy as np
import pandas as pd

from app import get_db
from src.db import fetch_summary, SORT_COLUMNS
from src.economics import (
    generate_expected_value_curve,
    select_threshold,
    THRESHOLD_GRID,
)


def parse_table_filters(filters: dict[str, str]) -> dict[str, Any]:
    """Converts table filters in the customer page into a Python dictionary for SQL selection filtering."""
    risk_band = filters.get("band")
    if risk_band not in {"High", "Medium", "Low"}:
        risk_band = None

    country = filters.get("country")
    if not country:
        country = None

    sort = filters.get("sort", "churn_proba")
    if sort not in SORT_COLUMNS:
        sort = "churn_proba"

    try:
        page = max(1, int(filters.get("page", 1)))
    except ValueError:
        page = 1

    try:
        max_rows_per_page = min(max(1, int(filters.get("per_page", 50))), 200)
    except ValueError:
        max_rows_per_page = 50

    return {
        "risk_band": risk_band,
        "country": country,
        "sort": sort,
        "descending": filters.get("direction", "desc") == "desc",
        "page": page,
        "max_rows_per_page": max_rows_per_page,
    }


def build_pagination(
    page: int, max_rows_per_page: int, total_rows: int
) -> dict[str, Any]:
    """Computes the pagination values for the page-control context for the customers table."""
    total_pages = max(1, math.ceil(total_rows / max_rows_per_page))
    page = min(page, total_pages)
    start = ((page - 1) * max_rows_per_page) + 1 if total_rows else 0
    end = min(page * max_rows_per_page, total_rows)

    return {
        "current_page": page,
        "total_pages": total_pages,
        "total_entries": total_rows,
        "page_entries_list_start": start,
        "page_entries_list_end": end,
        "has_prev_page": page > 1,
        "has_next_page": page < total_pages,
        "previous_page": page - 1,
        "next_page": page + 1,
    }


def population_comparison(
    customer: dict[str, Any], population: pd.DataFrame, features: list[str]
) -> list[dict[str, Any]]:
    """Determines a customer's percentile for every feature against the full population."""
    rows = []
    for feature in features:
        values = population[feature].dropna()
        customer_value = customer[feature]
        rows.append(
            {
                "feature": feature,
                "value": float(customer_value),
                "percentile": float((values <= customer_value).mean() * 100),
            }
        )
    return rows


def compute_economics(
    assumptions: dict[str, Any], y_true: np.ndarray, y_proba: np.ndarray
) -> dict[str, Any]:
    """Computes campaign economics under the given assumptions."""
    n_scored = fetch_summary(get_db())["customers_scored"]
    economics = parse_economics_parameters(assumptions)

    cost_per_contact = economics["cost_per_contact"]
    average_customer_lifetime_value = economics["average_customer_lifetime_value"]
    save_rate = economics["save_rate"]
    sensitivity_spread = current_app.config["ECONOMICS"].sensitivity_spread

    summary = select_threshold(
        y_true=y_true,
        y_proba=y_proba,
        n_scored=n_scored,
        cost=cost_per_contact,
        customer_lifetime_value=average_customer_lifetime_value,
        save_rate=save_rate,
        sensitivity_spread=sensitivity_spread,
        sensitivity_points=current_app.config["ECONOMICS"].sensitivity_points,
        capacity=economics["campaign_capacity"],
    )

    lower_bound = max(0.0, (save_rate - sensitivity_spread))
    upper_bound = min(max(0.0, (save_rate + sensitivity_spread)), 1.0)

    curve = generate_expected_value_curve(
        y_true=y_true,
        y_proba=y_proba,
        thresholds=THRESHOLD_GRID,
        save_rate=save_rate,
        customer_lifetime_value=average_customer_lifetime_value,
        cost=cost_per_contact,
    )

    return {
        **summary,
        "curve": {
            "thresholds": [row["threshold"] for row in curve],
            "expected_value": [row["expected_value"] for row in curve],
            "lower_bound": [
                (row["tp"] * lower_bound * average_customer_lifetime_value)
                - (row["n_flagged"] * cost_per_contact)
                for row in curve
            ],
            "upper_bound": [
                (row["tp"] * upper_bound * average_customer_lifetime_value)
                - (row["n_flagged"] * cost_per_contact)
                for row in curve
            ],
        },
    }


def parse_economics_parameters(parameters: dict[str, Any] | None) -> dict[str, float]:
    """
    Validates and coerces the campaign economics simulator parameters. Raises exceptions on anything
    missing, malformed or out of range.
    """
    if not parameters:
        raise ValueError("No campaign parameters were supplied.")

    cost_per_contact = float(parameters["cost_per_contact"])
    average_customer_lifetime_value = float(
        parameters["average_customer_lifetime_value"]
    )
    save_rate = float(parameters["save_rate"])
    campaign_capacity = int(parameters["campaign_capacity"])

    if not all(
        math.isfinite(value)
        for value in (cost_per_contact, average_customer_lifetime_value, save_rate)
    ):
        raise ValueError("Campaign parameters must be finite numbers.")

    if (
        (cost_per_contact <= 0)
        or (average_customer_lifetime_value <= 0)
        or (campaign_capacity < 1)
        or not (0 <= save_rate <= 1)
    ):
        raise ValueError("Campaign parameters were out of range.")

    return {
        "cost_per_contact": cost_per_contact,
        "average_customer_lifetime_value": average_customer_lifetime_value,
        "save_rate": save_rate,
        "campaign_capacity": campaign_capacity,
    }
