"""
Module containing all SQLite database functions.

Writing functions use the context-managed open_connection() function to manage their own scope. Reading
functions take a sqlite3.Connection argument to enable the Flask requests to share a single connection.
"""

from contextlib import contextmanager
from collections.abc import Generator
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customer_scores (
    customer_id INTEGER PRIMARY KEY,
    churn_proba REAL NOT NULL,
    risk_band TEXT NOT NULL,
    recency INTEGER,
    frequency INTEGER,
    monetary REAL,
    tenure INTEGER,
    country TEXT,
    predicted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scores_proba ON customer_scores(churn_proba DESC);
CREATE INDEX IF NOT EXISTS idx_scores_band ON customer_scores(risk_band);
"""

SORT_COLUMNS = {"churn_proba", "recency", "frequency", "monetary", "tenure", "country"}


def make_connection(db_path: Path) -> sqlite3.Connection:
    """Opens a database connection with row_factory set to sqlite3.Row."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def open_connection(db_path: Path) -> Generator[sqlite3.Connection]:
    """Opens a connection that commits SQLite queries on success and rolls back on failures."""
    conn = make_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    """Creates SQL tables and indexes (if non-existent)."""
    with open_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def write_scores(db_path: Path, scores: pd.DataFrame) -> int:
    """Replaces the contents of customer_scores with a fresh scoring pass. Returns the number of rows written."""
    columns = [
        "customer_id",
        "churn_proba",
        "risk_band",
        "recency",
        "frequency",
        "monetary",
        "tenure",
        "country",
        "predicted_at",
    ]
    insert_query = f"""
    INSERT INTO customer_scores ({", ".join(columns)}) 
    VALUES ({", ".join(f":{column}" for column in columns)})
    """
    records = scores[columns].to_dict(orient="records")

    with open_connection(db_path) as conn:
        conn.execute("DELETE FROM customer_scores")
        conn.executemany(insert_query, records)

    return len(records)


def fetch_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """
    Gets the KPIs (total customers scored, mean predicted churn probability, revenue at risk,
    and the timestamp of the scoring run) for the overview page.
    """
    row = conn.execute("""
        SELECT 
            COUNT(*) AS customers_scored,
            COALESCE(AVG(churn_proba), 0.0) AS mean_churn_proba,
            COALESCE(SUM(monetary * churn_proba), 0.0) AS revenue_at_risk,
            MAX(predicted_at) AS timestamp
        FROM customer_scores
    """).fetchone()

    return dict(row)


def fetch_band_distribution(conn: sqlite3.Connection) -> dict[str, Any]:
    """Gets the counts per risk band for the doughnut chart."""
    bands = conn.execute("""
        SELECT risk_band, COUNT(*) AS n
        FROM customer_scores
        GROUP BY risk_band
    """).fetchall()

    return {band["risk_band"]: band["n"] for band in bands}


def fetch_scores(
    conn: sqlite3.Connection,
    risk_band: str | None = None,
    country: str | None = None,
    sort: str = "churn_proba",
    descending: bool = True,
    page: int = 1,
    max_rows_per_page: int = 50,
) -> tuple[list[sqlite3.Row], int]:
    """
    Returns a single HTML page worth of scored rows for the customers table, alongside the
    total number of scored rows (after filtering by risk_band and/or country).
    """
    if sort not in SORT_COLUMNS:
        raise ValueError(
            f"value for 'sort' must be in {SORT_COLUMNS}; {sort} is not accepted"
        )

    order = "DESC" if descending else "ASC"

    clauses, params = [], {}
    if risk_band:
        clauses.append("risk_band = :risk_band")
        params["risk_band"] = risk_band
    if country:
        clauses.append("country = :country")
        params["country"] = country
    where_condition = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    total = conn.execute(
        f"""
        SELECT COUNT(*) AS total FROM customer_scores 
        {where_condition}
    """,
        params,
    ).fetchone()["total"]

    scores = conn.execute(
        f"""
        SELECT * FROM customer_scores
        {where_condition}
        ORDER BY {sort} {order}, customer_id
        LIMIT :limit OFFSET :offset
    """,
        {
            **params,
            "limit": max_rows_per_page,
            "offset": (page - 1) * max_rows_per_page,
        },
    ).fetchall()

    return scores, total


def fetch_country_options(conn: sqlite3.Connection) -> list[str]:
    """Gets the list of distinct countries in the scored set for the filter dropdown."""
    countries = conn.execute("""
        SELECT DISTINCT country FROM customer_scores
        WHERE country IS NOT NULL
        ORDER BY country
    """).fetchall()

    return [country["country"] for country in countries]


def fetch_customer(conn: sqlite3.Connection, customer_id: int) -> sqlite3.Row | None:
    """Gets a single customer's score row. Returns None if the customer_id is not present."""
    row = (
        conn.execute(
            "SELECT * FROM customer_scores WHERE customer_id = :customer_id",
            {"customer_id": customer_id},
        ).fetchone()
    )  # Cursor.fetchone() returns None if the input customer_id does not exist

    return row
