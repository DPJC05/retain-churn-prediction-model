"""Generates 'churned' labels for candidates in processed dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.features import filter_cancellations, filter_by_window


def get_eligible_customers(
    df: pd.DataFrame, cutoff: str, eligibility_lookback: int
) -> pd.DataFrame:
    """Identifies the 'customer_id' values of all eligible customers in the window."""
    eligible = filter_cancellations(filter_by_window(df, cutoff, eligibility_lookback))
    return pd.DataFrame(
        {
            "customer_id": np.sort(eligible["customer_id"].unique()),
            "cutoff": pd.Timestamp(cutoff),
        }
    )


def build_churn_label(
    df_labels: pd.DataFrame, df_interim: pd.DataFrame, cutoff: str, churn_horizon: int
) -> pd.DataFrame:
    """Calculates the purchase frequency per customer during the outcome window, then generates the 'churned' label."""
    df_modified = df_labels.copy()
    df_filtered = filter_by_outcome_window(df_interim, cutoff, churn_horizon)
    df_filtered = filter_cancellations(df_filtered)

    # Calculates the purchase frequency per customer during the outcome window
    orders_per_customer = (
        df_filtered.groupby(["customer_id", "invoice"])["invoice_date"]
        .min()
        .reset_index()
        .drop(columns="invoice_date")
    )
    outcome_frequency = (
        orders_per_customer.groupby("customer_id")["invoice"].count()
    ).rename("outcome_frequency")
    df_modified = pd.merge(
        df_modified,
        outcome_frequency,
        left_on="customer_id",
        right_index=True,
        how="left",
    )
    df_modified["outcome_frequency"] = df_modified["outcome_frequency"].fillna(
        0
    )  # All customers with an 'outcome_frequency' of 0 will be listed as NaN

    # Generates the 'churned' label
    df_modified["churned"] = (df_modified["outcome_frequency"] == 0).astype(int)

    # Drops temporary 'outcome_frequency' column
    df_modified = df_modified.drop(columns="outcome_frequency")

    return df_modified


def filter_by_outcome_window(
    df: pd.DataFrame, cutoff: str, churn_horizon: int
) -> pd.DataFrame:
    """Generates slice of interim dataset containing all transactions between the cutoff and churn_horizon."""
    cutoff_date = pd.Timestamp(cutoff)
    mask = df["invoice_date"] > cutoff_date
    mask &= df["invoice_date"] <= (cutoff_date + pd.Timedelta(days=churn_horizon))
    return df[mask]


def engineer_labels(
    df_interim: pd.DataFrame, cutoff: str, eligibility_lookback: int, churn_horizon: int
) -> pd.DataFrame:
    """Generates 'churned' labels for customers based on a single outcome window."""
    df_labels = get_eligible_customers(
        df=df_interim, cutoff=cutoff, eligibility_lookback=eligibility_lookback
    )
    df_labels = build_churn_label(
        df_labels=df_labels,
        df_interim=df_interim,
        cutoff=cutoff,
        churn_horizon=churn_horizon,
    )
    return df_labels


def label_customers(
    interim_path: Path,
    cutoffs: list[str],
    eligibility_lookback: int,
    churn_horizon: int,
    processed_path: Path,
) -> None:
    """Engine for generating 'churned' labels for all the customers in the input dataset."""

    print("\n --------------- labelling.py start ---------------")

    print("\n (1) Loading interim dataset into labelling.py")
    cutoff_dates = cutoffs.copy()
    df_interim = pd.read_parquet(interim_path)

    print("\n (2) Building labels")
    frames = [
        engineer_labels(df_interim, cutoff_date, eligibility_lookback, churn_horizon)
        for cutoff_date in cutoff_dates
    ]
    labels = pd.concat(frames, ignore_index=True)

    print("\n (3) Loading processed dataset into labelling.py")
    df_processed = pd.read_parquet(processed_path)

    print("\n (4) Adding labels to processed dataset")
    if "churned" in df_processed.columns:
        df_processed = df_processed.drop(columns="churned")
    df_processed = df_processed.merge(labels, on=["customer_id", "cutoff"], how="inner")
    assert len(df_processed) == len(labels), (
        "Merging of labels resulted in dropped or duplicated rows"
    )
    assert df_processed["churned"].notna().all(), "Labels contain NaN values"

    print(f"\n (5) Persisting processed dataset as {processed_path}")
    df_processed.to_parquet(processed_path, engine="pyarrow", index=False)

    print("\n ---------------- labelling.py end ----------------")


def main() -> None:
    """Runs label_customers() with default config values."""
    config = load_config()
    label_customers(
        interim_path=config.data.interim_path,
        cutoffs=config.labelling.cutoffs,
        churn_horizon=config.labelling.churn_horizon,
        eligibility_lookback=config.labelling.eligibility_lookback,
        processed_path=config.data.processed_path,
    )


if __name__ == "__main__":
    main()
