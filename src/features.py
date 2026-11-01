"""Transforms interim data into processed dataset via feature engineering."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config


def filter_by_window(
    df: pd.DataFrame, cutoff: str, eligibility_lookback: int | None = None
) -> pd.DataFrame:
    """
    Generates slice of interim dataset with all transactions after the cutoff and before the lookback dropped.
    If no eligibility_lookback argument is provided, the full history up to the cutoff is returned.
    """
    cutoff_date = pd.Timestamp(cutoff)
    mask = df["invoice_date"] <= cutoff_date
    if eligibility_lookback is not None:
        mask &= df["invoice_date"] >= (
            cutoff_date - pd.Timedelta(days=eligibility_lookback)
        )
    return df[mask]


def filter_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Generates slice of interim dataset with all cancelled transactions dropped."""
    return df[df["cancelled"] == 0]


def build_group_identifiers(
    df: pd.DataFrame, cutoff: str, eligibility_lookback: int, churn_horizon: int
) -> pd.DataFrame:
    """Generates the 'customer_id', 'cutoff', and 'churn_horizon' features."""
    eligible = filter_cancellations(filter_by_window(df, cutoff, eligibility_lookback))
    return pd.DataFrame(
        {
            "customer_id": np.sort(eligible["customer_id"].unique()),
            "cutoff": pd.Timestamp(cutoff),
            "churn_horizon": churn_horizon,
        }
    )


def build_rfm_features(
    df_processed: pd.DataFrame, df_interim: pd.DataFrame, cutoff: str
) -> pd.DataFrame:
    """Generates the RFM-related features."""
    df_modified = df_processed.copy()
    df_filtered = filter_by_window(df_interim, cutoff, None)
    df_filtered = filter_cancellations(df_filtered)

    # Generates the 'recency' feature
    recency = (
        pd.Timestamp(cutoff).normalize()
        - df_filtered.groupby("customer_id")["invoice_date"].max().dt.normalize()
    ).dt.days.rename("recency")
    df_modified = pd.merge(
        df_modified, recency, left_on="customer_id", right_index=True, how="left"
    )

    # Generates the 'frequency' feature
    orders_per_customer = (
        df_filtered.groupby(["customer_id", "invoice"])["invoice_date"]
        .min()
        .reset_index()
        .drop(columns="invoice_date")
    )
    frequency = (orders_per_customer.groupby("customer_id")["invoice"].count()).rename(
        "frequency"
    )
    df_modified = pd.merge(
        df_modified, frequency, left_on="customer_id", right_index=True, how="left"
    )

    # Generates the 'monetary' feature (gross revenue per customer)
    monetary = (df_filtered.groupby("customer_id")["line_total"].sum()).rename(
        "monetary"
    )
    df_modified = pd.merge(
        df_modified, monetary, left_on="customer_id", right_index=True, how="left"
    )

    # Generates the 'mean_order_value' feature
    df_modified["mean_order_value"] = df_modified["monetary"] / df_modified["frequency"]

    return df_modified


def build_tenure_feature(
    df_processed: pd.DataFrame, df_interim: pd.DataFrame, cutoff: str
) -> pd.DataFrame:
    """Generates the 'tenure' feature."""
    df_modified = df_processed.copy()
    df_filtered = filter_by_window(df_interim, cutoff, None)
    df_filtered = filter_cancellations(df_filtered)

    first_purchase = df_filtered.groupby("customer_id")["invoice_date"].min().to_frame()
    tenure = (
        pd.Timestamp(cutoff).normalize() - first_purchase["invoice_date"].dt.normalize()
    ).dt.days.rename("tenure")
    df_modified = pd.merge(
        df_modified, tenure, left_on="customer_id", right_index=True, how="left"
    )

    return df_modified


def build_interval_features(
    df_processed: pd.DataFrame, df_interim: pd.DataFrame, cutoff: str
) -> pd.DataFrame:
    """Generates the purchase interval-related features."""
    df_modified = df_processed.copy()
    df_filtered = filter_by_window(df_interim, cutoff, None)
    df_filtered = filter_cancellations(df_filtered)

    # Generates the 'avg_inter_purchase_gap' and 'std_inter_purchase_gap' features
    orders = (
        df_filtered.groupby(["customer_id", "invoice"])["invoice_date"]
        .min()
        .reset_index()
        .sort_values(["customer_id", "invoice_date"])
    )
    orders["days_since_prev_order"] = (
        orders.groupby("customer_id")["invoice_date"].diff().dt.days
    )
    intervals = orders.groupby("customer_id")["days_since_prev_order"].agg(
        avg_inter_purchase_gap="mean",
        std_inter_purchase_gap="std",
    )
    df_modified = pd.merge(
        df_modified, intervals, left_on="customer_id", right_index=True, how="left"
    )

    # Generates the 'recency_ratio' feature
    df_modified["recency_ratio"] = np.where(
        df_modified["avg_inter_purchase_gap"] > 0,
        df_modified["recency"] / df_modified["avg_inter_purchase_gap"],
        np.nan,
    )

    return df_modified


def build_activity_features(
    df_processed: pd.DataFrame,
    df_interim: pd.DataFrame,
    cutoff: str,
    lookback_days: list[int],
) -> pd.DataFrame:
    """Generates the features related to order and spending activity."""
    df_modified = df_processed.copy()
    df_filtered = filter_by_window(df_interim, cutoff)
    df_filtered = filter_cancellations(df_filtered)
    df_filtered_earliest = filter_by_window(df_filtered, cutoff, lookback_days[0])
    df_filtered_middle = filter_by_window(df_filtered, cutoff, lookback_days[1])
    df_filtered_latest = filter_by_window(df_filtered, cutoff, lookback_days[2])
    frames = [df_filtered_earliest, df_filtered_middle, df_filtered_latest]

    # Generates the 'spend_{earliest}d', 'spend_{middle}d' and 'spend_{latest}d' features
    spend_features = [
        build_spend_feature(frame, day) for frame, day in zip(frames, lookback_days)
    ]
    spend_features = pd.concat(spend_features, axis=1, join="outer")
    df_modified = pd.merge(
        df_modified, spend_features, left_on="customer_id", right_index=True, how="left"
    )

    # Generates the 'orders_{earliest}d', 'orders_{middle}d' and 'orders_{latest}d' features
    orders_features = [
        build_orders_feature(frame, day) for frame, day in zip(frames, lookback_days)
    ]
    orders_features = pd.concat(orders_features, axis=1, join="outer")
    df_modified = pd.merge(
        df_modified,
        orders_features,
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    # Fills all NaN values in 'spend_Xd' and 'orders_Xd' features
    columns = [
        f"spend_{lookback_days[0]}d",
        f"spend_{lookback_days[1]}d",
        f"spend_{lookback_days[2]}d",
        f"orders_{lookback_days[0]}d",
        f"orders_{lookback_days[1]}d",
        f"orders_{lookback_days[2]}d",
    ]
    df_modified[columns] = df_modified[columns].fillna(0)

    # Generates the 'spend_momentum_{earliest}d_{middle}d' and 'spend_momentum_{middle}d_{latest}d' features
    df_modified[f"spend_momentum_{lookback_days[0]}d_{lookback_days[1]}d"] = np.where(
        df_modified[f"spend_{lookback_days[1]}d"] > 0,
        df_modified[f"spend_{lookback_days[0]}d"]
        / (
            df_modified[f"spend_{lookback_days[1]}d"]
            / (lookback_days[1] / lookback_days[0])
        ),  # Divided by ratio of middle:earliest to scale into a comparable rate
        np.nan,
    )
    df_modified[f"spend_momentum_{lookback_days[1]}d_{lookback_days[2]}d"] = np.where(
        df_modified[f"spend_{lookback_days[2]}d"] > 0,
        df_modified[f"spend_{lookback_days[1]}d"]
        / (
            df_modified[f"spend_{lookback_days[2]}d"]
            * (lookback_days[1] / lookback_days[2])
        ),  # Multiplied by ratio of middle:latest to scale into a comparable rate
        np.nan,
    )

    # Generates the 'order_momentum_{middle}d_{latest}d' and 'order_momentum_{middle}d_{latest}d' features
    df_modified[f"order_momentum_{lookback_days[0]}d_{lookback_days[1]}d"] = np.where(
        df_modified[f"orders_{lookback_days[1]}d"] > 0,
        df_modified[f"orders_{lookback_days[0]}d"]
        / (
            df_modified[f"orders_{lookback_days[1]}d"]
            / (lookback_days[1] / lookback_days[0])
        ),  # Divided by ratio of middle:earliest to scale into a comparable rate
        np.nan,
    )
    df_modified[f"order_momentum_{lookback_days[1]}d_{lookback_days[2]}d"] = np.where(
        df_modified[f"orders_{lookback_days[2]}d"] > 0,
        df_modified[f"orders_{lookback_days[1]}d"]
        / (
            df_modified[f"orders_{lookback_days[2]}d"]
            * (lookback_days[1] / lookback_days[2])
        ),  # Multiplied by ratio of middle:latest to scale into a comparable rate
        np.nan,
    )

    return df_modified


def build_spend_feature(df: pd.DataFrame, days: int) -> pd.Series:
    """Helper function for build_activity_features(). Generates a single 'spend_Xd' feature."""
    series_name = f"spend_{days}d"
    return (df.groupby("customer_id")["line_total"].sum()).rename(series_name)


def build_orders_feature(df: pd.DataFrame, days: int) -> pd.Series:
    """Helper function for build_activity_features(). Generates a single 'orders_Xd' feature."""
    series_name = f"orders_{days}d"
    orders_per_customer = (
        df.groupby(["customer_id", "invoice"])["invoice_date"]
        .min()
        .reset_index()
        .drop(columns="invoice_date")
    )
    return (orders_per_customer.groupby("customer_id")["invoice"].count()).rename(
        series_name
    )


def build_distinct_products_feature(
    df_processed: pd.DataFrame, df_interim: pd.DataFrame, cutoff: str
) -> pd.DataFrame:
    """Generates the 'distinct_products' feature."""
    df_modified = df_processed.copy()
    df_filtered = filter_by_window(df_interim, cutoff, None)
    df_filtered = filter_cancellations(df_filtered)

    distinct_products = (
        df_filtered.groupby("customer_id")["stock_code"]
        .nunique()
        .rename("distinct_products")
    )
    df_modified = pd.merge(
        df_modified,
        distinct_products,
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    return df_modified


def build_mean_order_quantity_feature(
    df_processed: pd.DataFrame, df_interim: pd.DataFrame, cutoff: str
) -> pd.DataFrame:
    """Generates the 'mean_order_quantity' feature."""
    df_modified = df_processed.copy()
    df_filtered = filter_by_window(df_interim, cutoff, None)
    df_filtered = filter_cancellations(df_filtered)

    total_order_quantities = (
        df_filtered.groupby(["customer_id", "invoice"])["quantity"].sum().to_frame()
    )
    mean_order_quantities = (
        total_order_quantities.groupby("customer_id")["quantity"]
        .mean()
        .rename("mean_order_quantity")
    )
    df_modified = pd.merge(
        df_modified,
        mean_order_quantities,
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    return df_modified


def build_return_rate_feature(
    df_processed: pd.DataFrame,
    df_interim: pd.DataFrame,
    cutoff: str,
) -> pd.DataFrame:
    """Generates the 'return_rate' feature."""
    df_modified = df_processed.copy()
    df_filtered = filter_by_window(df_interim, cutoff, None)

    total_transactions = (
        df_filtered.groupby("customer_id")["invoice"]
        .nunique()
        .rename("total_transactions")
    )
    total_returns = (
        df_filtered[df_filtered["cancelled"] == 1]
        .groupby("customer_id")["invoice"]
        .nunique()
        .rename("total_returns")
    )
    tmp = pd.merge(
        total_transactions,
        total_returns,
        left_on="customer_id",
        right_index=True,
        how="left",
    )
    tmp["total_returns"] = tmp["total_returns"].fillna(0)
    tmp["return_rate"] = tmp["total_returns"] / tmp["total_transactions"]
    df_modified = pd.merge(
        df_modified,
        tmp["return_rate"],
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    return df_modified


def build_country_feature(
    df_processed: pd.DataFrame, df_interim: pd.DataFrame, cutoff: str
) -> pd.DataFrame:
    """Generates the 'country' feature."""
    df_modified = df_processed.copy()
    df_filtered = filter_by_window(df_interim, cutoff, None)
    df_filtered = filter_cancellations(df_filtered)

    countries = (
        (df_filtered.groupby(["customer_id", "invoice"])["country"])
        .first()
        .groupby(level="customer_id")
        .agg(
            lambda countries_subseries: countries_subseries.mode().iloc[
                0
            ]  # To handle potential ties for multi-country customers
        )
        .rename("country")
    )
    df_modified = pd.merge(
        df_modified, countries, left_on="customer_id", right_index=True, how="left"
    )

    return df_modified


def engineer_features(
    df_interim: pd.DataFrame,
    cutoff: str,
    churn_horizon: int,
    eligibility_lookback: int,
    lookback_days: list[int],
) -> pd.DataFrame:
    """Engine for transforming the interim data into its processed form for a single cutoff."""
    df_processed = build_group_identifiers(
        df=df_interim,
        cutoff=cutoff,
        churn_horizon=churn_horizon,
        eligibility_lookback=eligibility_lookback,
    )
    df_processed = build_rfm_features(
        df_processed=df_processed, df_interim=df_interim, cutoff=cutoff
    )
    df_processed = build_tenure_feature(
        df_processed=df_processed, df_interim=df_interim, cutoff=cutoff
    )
    df_processed = build_interval_features(
        df_processed=df_processed, df_interim=df_interim, cutoff=cutoff
    )
    df_processed = build_activity_features(
        df_processed=df_processed,
        df_interim=df_interim,
        cutoff=cutoff,
        lookback_days=lookback_days,
    )
    df_processed = build_distinct_products_feature(
        df_processed=df_processed, df_interim=df_interim, cutoff=cutoff
    )
    df_processed = build_mean_order_quantity_feature(
        df_processed=df_processed, df_interim=df_interim, cutoff=cutoff
    )
    df_processed = build_return_rate_feature(
        df_processed=df_processed, df_interim=df_interim, cutoff=cutoff
    )
    df_processed = build_country_feature(
        df_processed=df_processed, df_interim=df_interim, cutoff=cutoff
    )
    assert len(df_processed) == df_processed["customer_id"].nunique(), (
        "Duplicate rows found; merge required."
    )
    return df_processed


def build_dataset(
    interim_path: Path,
    cutoffs: list[str],
    churn_horizon: int,
    eligibility_lookback: int,
    lookback_days: list[int],
    processed_path: Path,
) -> None:
    """Engine for running engineer_features() for all cutoffs."""

    print("\n --------------- features.py start ----------------")

    print("\n (1) Loading interim dataset into features.py")
    cutoff_dates = cutoffs.copy()
    df_interim = pd.read_parquet(interim_path)

    print("\n (2) Engineering features")
    frames = [
        engineer_features(
            df_interim, cutoff_date, churn_horizon, eligibility_lookback, lookback_days
        )
        for cutoff_date in cutoff_dates
    ]
    df = pd.concat(frames, ignore_index=True)

    df["country"] = df["country"].astype("category")

    print(f"\n (3) Persisting processed dataset as {processed_path}")
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, engine="pyarrow", index=False)

    print("\n ---------------- features.py end -----------------")


def main() -> None:
    """Runs build_dataset() with default config values."""
    config = load_config()
    build_dataset(
        interim_path=config.data.interim_path,
        cutoffs=config.labelling.cutoffs,
        churn_horizon=config.labelling.churn_horizon,
        eligibility_lookback=config.labelling.eligibility_lookback,
        lookback_days=config.labelling.lookback_days,
        processed_path=config.data.processed_path,
    )


if __name__ == "__main__":
    main()
