"""Cleans raw dataset and prepares it for labelling and feature engineering."""

from pathlib import Path
import json

import pandas as pd

from src.config import load_config


def load_data(raw_path: Path) -> pd.DataFrame:
    """Imports raw dataset from 'data/raw/' directory as a Pandas DataFrame."""
    sheets = pd.read_excel(raw_path, sheet_name=["Year 2009-2010", "Year 2010-2011"])
    return pd.concat(sheets.values(), ignore_index=True)


def standardise_column_headers(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Converts all column headers to snake_case."""
    df_modified = df.copy()
    df_modified.columns = df_modified.columns.str.strip()
    return df_modified.rename(columns=column_map)


def drop_duplicate_entries(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drops all duplicate line item purchases."""
    dropped = df.duplicated().sum()
    return df.drop_duplicates(ignore_index=True), int(dropped)


def get_identified_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drops all line item purchases with NaN values in customer_id column."""
    dropped = df["customer_id"].isna().sum()
    return df.dropna(
        axis=0, how="any", subset="customer_id", inplace=False, ignore_index=True
    ), int(dropped)


def drop_non_products(
    df: pd.DataFrame, non_products: list[str]
) -> tuple[pd.DataFrame, int]:
    """Drops all non-product line item purchases."""
    stock_codes = df["stock_code"].str.strip().str.upper()
    stock_codes_dropped = stock_codes.isin(
        {item.strip().upper() for item in non_products}
    )
    dropped = (stock_codes_dropped).sum()
    return df[~stock_codes_dropped].copy().reset_index(drop=True), int(dropped)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the following columns are set to their corresponding dtypes:
    1. invoice (string)
    2. stock_code (string)
    3. description (string)
    4. quantity (int32)
    5. invoice_date (datetime64[us])
    6. price (float64)
    7. customer_id (int64)
    8. country (string)
    """
    df_modified = df.copy()
    df_modified["invoice"] = df_modified["invoice"].astype("string")
    df_modified["stock_code"] = df_modified["stock_code"].astype("string")
    df_modified["description"] = df_modified["description"].astype("string")
    df_modified["quantity"] = df_modified["quantity"].astype("int32")
    df_modified["invoice_date"] = pd.to_datetime(df_modified["invoice_date"]).astype(
        "datetime64[us]"
    )
    df_modified["price"] = df_modified["price"].astype("float64")
    df_modified["customer_id"] = df_modified["customer_id"].astype("int64")
    df_modified["country"] = df_modified["country"].astype("string")
    return df_modified


def flag_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Inserts a new 'cancelled' column to flags all cancelled line item purchases."""
    df_modified = df.copy()
    df_modified["cancelled"] = (
        df_modified["invoice"].str.startswith("C").fillna(False).astype(int)
    )
    return df_modified


def get_line_total(df: pd.DataFrame) -> pd.DataFrame:
    """Inserts a new 'line_total' column to combine 'price' with 'quantity' per line item purchase."""
    df_modified = df.copy()
    df_modified["line_total"] = df_modified["price"] * df_modified["quantity"]
    return df_modified


def prepare_data(
    raw_path: Path,
    column_map: dict,
    non_products: list[str],
    interim_path: Path,
    purge_report_path: Path,
) -> None:
    """Engine for transforming the raw dataset into its interim preprocessed form. Reports data deletions as JSON."""

    print("\n --------------- data_prep.py start ---------------")

    print("\n (1) Loading raw dataset into data_prep.py")
    df = load_data(raw_path)
    raw_count = len(df)

    print("\n (2) Cleaning raw dataset")

    df = standardise_column_headers(df, column_map)
    df, count1 = drop_duplicate_entries(df)
    df, count2 = get_identified_customers(df)
    df, count3 = drop_non_products(df, non_products)
    df = coerce_types(df)
    df = flag_cancellations(df)
    df = get_line_total(df)
    df = df.sort_values(["customer_id", "invoice_date"]).reset_index(drop=True)
    final_count = len(df)

    print(f"\n (3) Purge report for data_prep.py")
    counts = {
        "initial_rows": raw_count,
        "duplicates_dropped": count1,
        "unidentified_dropped": count2,
        "non_products_dropped": count3,
        "final_rows": final_count,
    }
    print(f"\n{counts}")

    print(f"\n (4) Persisting processed dataset as {interim_path}")
    Path(interim_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_path, engine="pyarrow", index=False)

    print(f"\n (5) Persisting purge report as {purge_report_path}")
    Path(purge_report_path).parent.mkdir(exist_ok=True)
    Path(purge_report_path).write_text(json.dumps(counts, indent=2))

    print("\n ---------------- data_prep.py end ----------------")


def main() -> None:
    """Runs prepare_data() with default config values."""
    config = load_config()
    prepare_data(
        raw_path=config.data.raw_path,
        column_map=config.prep.column_map,
        non_products=config.prep.non_products,
        interim_path=config.data.interim_path,
        purge_report_path=config.prep.purge_report_persist_path,
    )


if __name__ == "__main__":
    main()
