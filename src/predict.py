"""
Predicts churn probability for customers with cutoff set as the latest transaction date, assigns them
with churn risk bands, and writes the full list of scored customers to a SQLite database.
"""

from datetime import datetime
import joblib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.features import engineer_features
from src.db import init_db, write_scores


def load_artifact(
    model_persist_path: Path, metadata_persist_path: Path
) -> tuple[Pipeline, dict[str, Any]]:
    """Loads the trained model and metadata."""
    model = joblib.load(model_persist_path)
    metadata = json.loads(metadata_persist_path.read_text())
    return model, metadata


def assign_risk_band(
    churn_proba: np.ndarray,
    metadata: dict[str, Any],
    medium_risk_band_multiplier: float,
) -> np.ndarray:
    """Assigns 'High' / 'Medium' / 'Low' risk bands to each customer based on churn probability."""
    scoring_thresholds = {
        "high": metadata["chosen_threshold"],
        "medium": metadata["chosen_threshold"] * medium_risk_band_multiplier,
    }
    risk_bands = np.where(
        churn_proba >= scoring_thresholds["medium"],
        np.where(churn_proba >= scoring_thresholds["high"], "High", "Medium"),
        "Low",
    )
    return risk_bands


def build_score_records(
    scoring_frame: pd.DataFrame,
    churn_proba: np.ndarray,
    risk_bands: np.ndarray,
    predicted_at: str,
) -> pd.DataFrame:
    """Constructs the customer scores table."""
    df = scoring_frame.copy()
    df["churn_proba"] = churn_proba
    df["risk_band"] = risk_bands.tolist()
    df["predicted_at"] = predicted_at
    df["customer_id"] = df["customer_id"].astype(int)
    return df


def score_all(
    model_persist_path: Path,
    metadata_persist_path: Path,
    interim_path: Path,
    eligibility_lookback: int,
    lookback_days: list[int],
    medium_risk_band_multiplier: float,
    db_path: Path,
) -> None:
    """Engine for scoring all the eligible customers at the cutoff, and writing the scores to SQLite."""

    print("\n ---------------- predict.py start ----------------")

    print("\n (1) Loading artifacts and interim dataset into predict.py")
    model, metadata = load_artifact(model_persist_path, metadata_persist_path)
    df_interim = pd.read_parquet(interim_path)

    print("\n (2) Building a scoring frame for all eligible customers")
    scoring_frame = engineer_features(
        df_interim=df_interim,
        cutoff=df_interim["invoice_date"].max(),
        churn_horizon=metadata["churn_horizon"],
        eligibility_lookback=eligibility_lookback,
        lookback_days=lookback_days,
    )
    X = scoring_frame[metadata["feature_columns"]]

    print("\n (3) Predicting churn probability using the model")
    churn_proba = model.predict_proba(X)[:, 1]

    print("\n (4) Assigning risk bands to customers based on their churn probability")
    risk_bands = assign_risk_band(
        churn_proba=churn_proba,
        metadata=metadata,
        medium_risk_band_multiplier=medium_risk_band_multiplier,
    )

    print(
        "\n (5) Compiling features, predictions and risk bands into a score records table\n"
    )
    scores = build_score_records(
        scoring_frame=scoring_frame,
        churn_proba=churn_proba,
        risk_bands=risk_bands,
        predicted_at=datetime.now().isoformat(),
    )
    scores.info()

    print("\n (6) Writing score records table to SQLite database")
    init_db(db_path)
    rows = write_scores(db_path, scores)
    print(f"\n {rows} rows have been written to {db_path}")

    print("\n ----------------- predict.py end -----------------")


def main() -> None:
    """Runs score_all() with default config values."""
    config = load_config()
    score_all(
        model_persist_path=config.model.model_persist_path,
        metadata_persist_path=config.model.metadata_persist_path,
        interim_path=config.data.interim_path,
        eligibility_lookback=config.labelling.eligibility_lookback,
        lookback_days=config.labelling.lookback_days,
        medium_risk_band_multiplier=config.prediction.medium_risk_band_multiplier,
        db_path=config.database.db_path,
    )


if __name__ == "__main__":
    main()
