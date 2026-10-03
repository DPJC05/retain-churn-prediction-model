"""Renders the HTML templates to the browser."""

from collections.abc import Generator
import csv
import io

from flask import abort, Blueprint, current_app, Response, render_template, request
import pandas as pd

from app import get_db
from app.helpers import (
    parse_table_filters,
    build_pagination,
    population_comparison,
    compute_economics,
)
from src.db import (
    fetch_summary,
    fetch_band_distribution,
    fetch_scores,
    fetch_customer,
    fetch_country_options,
)


bp = Blueprint("main", __name__)


@bp.route("/")
def index() -> str:
    """Renders the overview page: headline KPIs and the risk distribution."""
    conn = get_db()
    kpis = dict(fetch_summary(conn))
    band_distribution = fetch_band_distribution(conn)
    provenance = current_app.config["METADATA"]
    return render_template(
        "index.html",
        kpis=kpis,
        band_distribution=band_distribution,
        provenance=provenance,
    )


@bp.route("/customers")
def customers() -> str:
    """Renders the paginated, sortable, filterable table of scored customers."""
    conn = get_db()
    filters = parse_table_filters(request.args)
    scores, total = fetch_scores(conn=conn, **filters)
    scores = [dict(row) for row in scores]
    countries = fetch_country_options(conn=conn)
    sort = [
        ("churn_proba", "Churn Risk"),
        ("recency", "Days Since Last Purchase"),
        ("frequency", "Order Count"),
        ("monetary", "Total Spend"),
        ("tenure", "Tenure"),
        ("country", "Country"),
    ]
    pagination = build_pagination(
        page=filters["page"],
        max_rows_per_page=filters["max_rows_per_page"],
        total_rows=total,
    )
    return render_template(
        "customers.html",
        scores=scores,
        bands=["High", "Medium", "Low"],
        countries=countries,
        sort=sort,
        current_filters=filters,
        pagination=pagination,
    )


@bp.route("/customers/export")
def export_customers() -> Response:
    """Streams the current filtered target list into a download CSV."""
    filters = parse_table_filters(request.args)
    filters["page"] = 1
    filters["max_rows_per_page"] = 100000
    rows, _ = fetch_scores(get_db(), **filters)

    def csv_generator() -> Generator:
        """
        Creates a Generator object to be used by the Response object returned
        by export_customers() to write the CSV file 1 row at a time.

        The writing/yielding sequence is as follows:
        1. Write CSV headers row to buffer
        2. Yield buffer
        3. Loop begins:
           a. Clear the buffer
           b. Write one row of data
           c. Yield the row
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "customer_id",
                "country",
                "recency",
                "frequency",
                "monetary",
                "tenure",
                "churn_proba",
                "risk_band",
                "predicted_at",
            ]
        )
        yield buffer.getvalue()
        for row in rows:
            buffer.seek(0)
            buffer.truncate()
            writer.writerow(
                [
                    row["customer_id"],
                    row["country"],
                    row["recency"],
                    row["frequency"],
                    f"{row['monetary']:.2f}",
                    row["tenure"],
                    f"{row['churn_proba']:.4f}",
                    row["risk_band"],
                    row["predicted_at"],
                ]
            )
            yield buffer.getvalue()

    return Response(
        csv_generator(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=retain_targets.csv"},
    )


@bp.route("/customers/<int:customer_id>")
def customer_details(customer_id: int) -> str:
    """Renders a single customer's score details page."""
    conn = get_db()
    customer_score = fetch_customer(conn=conn, customer_id=customer_id)
    if customer_score is None:
        abort(404, description="Customer Not Found")
    customer_score = dict(customer_score)
    kpis = dict(fetch_summary(conn))
    population = pd.read_sql_query(
        "SELECT recency, frequency, monetary, tenure FROM customer_scores", conn
    )
    percentiles = population_comparison(
        customer=customer_score,
        population=population,
        features=["recency", "frequency", "monetary", "tenure"],
    )
    return render_template(
        "customer_details.html",
        customer_score=customer_score,
        kpis=kpis,
        percentiles=percentiles,
    )


@bp.route("/performance")
def performance() -> str:
    """Renders the model performance page."""
    report = current_app.config["EVALUATION_REPORT"]
    test_count = report["split_report"].get("df_test_customers")
    provenance = current_app.config["METADATA"]
    evaluation = report["evaluation_table"]
    curve_data = current_app.config["CURVE_DATA"]
    return render_template(
        "performance.html",
        provenance=provenance,
        test_count=test_count,
        evaluation=evaluation,
        curve_data=curve_data,
    )


@bp.route("/economics")
def economics() -> str:
    """Renders the campaign simulator page."""
    y_true = current_app.config["Y_TRUE"]
    y_proba = current_app.config["Y_PROBA"]
    economics = current_app.config["ECONOMICS"]
    report = current_app.config["EVALUATION_REPORT"]
    threshold = report["threshold"]
    assumptions = threshold["assumptions"]
    try:
        rendered = compute_economics(assumptions, y_true, y_proba)
    except (ValueError, TypeError, KeyError):
        abort(400)
    return render_template(
        "economics.html",
        assumptions=assumptions,
        rendered=rendered,
        economics=economics,
    )
