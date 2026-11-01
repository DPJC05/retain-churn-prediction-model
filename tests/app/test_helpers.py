"""Unit tests for the app.helpers module."""

import numpy as np
import pandas as pd
import pytest

from app.helpers import (
    parse_table_filters,
    build_pagination,
    population_comparison,
    parse_economics_parameters,
)


def test_parse_table_filters():
    filters1 = {
        "band": "High",
        "country": "United Kingdom",
        "sort": "recency",
        "direction": "desc",
        "page": 30,
        "per_page": 30,
    }
    filters2 = {
        "band": "Churned",
        "country": "",
        "sort": "band",
        "direction": "asc",
        "page": "A-1",
        "per_page": "alot",
    }
    test1 = parse_table_filters(filters1)
    test2 = parse_table_filters(filters2)
    assert test1 == {
        "risk_band": "High",
        "country": "United Kingdom",
        "sort": "recency",
        "descending": True,
        "page": 30,
        "max_rows_per_page": 30,
    }
    assert test2 == {
        "risk_band": None,
        "country": None,
        "sort": "churn_proba",
        "descending": False,
        "page": 1,
        "max_rows_per_page": 50,
    }


def test_build_pagination():
    test = build_pagination(
        page=1,
        max_rows_per_page=50,
        total_rows=300,
    )
    assert test == {
        "current_page": 1,
        "total_pages": 6,
        "total_entries": 300,
        "page_entries_list_start": 1,
        "page_entries_list_end": 50,
        "has_prev_page": False,
        "has_next_page": True,
        "previous_page": 0,
        "next_page": 2,
    }


def test_population_comparison():
    customer = {
        "churn_risk": 0.8,
    }
    population = pd.DataFrame(
        {"churn_risk": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    )
    features = ["churn_risk"]
    test = population_comparison(customer, population, features)
    assert test == [{"feature": "churn_risk", "value": 0.8, "percentile": 80.0}]


def test_parse_economics_parameters():
    parameters1 = {
        "cost_per_contact": 3.50,
        "average_customer_lifetime_value": 250,
        "save_rate": 0.15,
        "campaign_capacity": 200,
    }
    parameters2 = {
        "cost_per_contact": np.inf,
        "average_customer_lifetime_value": np.inf,
        "save_rate": np.inf,
        "campaign_capacity": np.inf,
    }
    parameters3 = {
        "cost_per_contact": -1,
        "average_customer_lifetime_value": -1,
        "save_rate": -1,
        "campaign_capacity": 0,
    }
    test1 = parse_economics_parameters(parameters1)

    with pytest.raises(ValueError):
        parse_economics_parameters(parameters="")
        parse_economics_parameters(parameters2)
        parse_economics_parameters(parameters3)

    assert test1 == {
        "cost_per_contact": 3.50,
        "average_customer_lifetime_value": 250,
        "save_rate": 0.15,
        "campaign_capacity": 200,
    }
