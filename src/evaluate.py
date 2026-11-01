"""Generates evaluative metrics for churn prediction model performance."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.economics import confusion_counts, scale_capacity

SCORE_TYPES = ["probability", "ranking", "binary"]

METRIC_KEYS = [
    "kind",
    "score_type",
    "split",
    "n_rows",
    "n_positive",
    "base_rate",
    "roc_auc",
    "pr_auc",
    "brier",
    "mean_predicted",
    "threshold",
    "n_flagged",
    "precision",
    "recall",
    "f1",
    "lift",
    "note",
]


def build_row(
    y_true: np.ndarray,
    scores: np.ndarray,
    kind: str,
    split: str,
    score_type: str,
    threshold: float | None,
    n_flagged_cap: int | None,
    note: str | None,
) -> dict[str, Any]:
    """Helper function for all evaluation functions. Builds a single row of metrics."""
    if score_type not in SCORE_TYPES:
        raise ValueError(
            f"score_type must be one of {SCORE_TYPES}; {score_type} is not accepted"
        )

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if y_true.shape != scores.shape:
        raise ValueError("shape mismatch between y_true and scores")

    n_rows = int(y_true.size)
    n_positive = int(y_true.sum())
    base_rate = n_positive / n_rows if n_rows else 0.0

    single_class = n_positive in (0, n_rows)
    ranked = score_type in ("probability", "ranking")
    calibrated = score_type == "probability"

    row = dict.fromkeys(METRIC_KEYS)
    row.update(
        {
            "kind": kind,
            "score_type": score_type,
            "split": split,
            "n_rows": n_rows,
            "n_positive": n_positive,
            "base_rate": base_rate,
            "roc_auc": float(roc_auc_score(y_true, scores))
            if ranked and not single_class
            else None,
            "pr_auc": float(average_precision_score(y_true, scores))
            if ranked and not single_class
            else None,
            "brier": float(brier_score_loss(y_true, scores))
            if calibrated and n_rows
            else None,
            "mean_predicted": float(scores.mean()) if calibrated and n_rows else None,
            "note": note,
        }
    )

    if threshold is None and n_flagged_cap is None:
        return row

    if n_flagged_cap is not None:
        k = min(int(n_flagged_cap), n_rows)
        if k == 0:
            return row
        top_k = np.argsort(scores, kind="stable")[-k:]
        row["threshold"] = float(scores[top_k].min())

    else:
        row["threshold"] = float(threshold)

    counts = confusion_counts(y_true=y_true, y_proba=scores, threshold=row["threshold"])

    precision = (
        counts["tp"] / (counts["tp"] + counts["fp"])
        if (counts["tp"] + counts["fp"])
        else 0.0
    )
    recall = (
        counts["tp"] / (counts["tp"] + counts["fn"])
        if (counts["tp"] + counts["fn"])
        else 0.0
    )

    row.update(
        {
            "n_flagged": counts["tp"] + counts["fp"],
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0,
            "lift": precision / base_rate if base_rate else None,
        }
    )
    return row


def evaluate_baselines(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    capacity: int,
    random_state: int,
    heuristic_baseline: int,
    n_scored: int,
) -> pd.DataFrame:
    """Evaluates baselines to generate reference scores for evaluating model performance."""
    y_test = np.asarray(y_test).astype(int)

    k = scale_capacity(
        capacity=capacity, n_scored=n_scored, n_evaluated=int(y_test.size)
    )

    rows = [
        build_row(
            y_true=y_test,
            scores=np.random.default_rng(random_state).random(len(X_test)),
            kind="random",
            split="test",
            score_type="ranking",
            threshold=None,
            n_flagged_cap=k,
            note="uniform noise, expect roc_auc near 0.50",
        ),
        build_row(
            y_true=y_test,
            scores=X_test["recency"].to_numpy(dtype=float),
            kind="rule",
            split="test",
            score_type="ranking",
            threshold=None,
            n_flagged_cap=k,
            note="recency as a continuous ranking",
        ),
        build_row(
            y_true=y_test,
            scores=(X_test["recency"] > heuristic_baseline).to_numpy(dtype=float),
            kind="rule",
            split="test",
            score_type="binary",
            threshold=0.5,
            n_flagged_cap=None,
            note=f"recency_above_{heuristic_baseline}d; hard 0/1, ranking metrics not defined",
        ),
    ]
    index = [
        "Random",
        "Recency (Ranked)",
        f"Recency Above {heuristic_baseline}d (Heuristic)",
    ]
    return pd.DataFrame(rows, index=index)


def generate_cv_metrics(search, y_train: pd.Series) -> dict[str, Any]:
    """Extracts the winning row from search.cv_results_ as a dictionary using the comparison schema."""
    y_train = np.asarray(y_train).astype(int)

    results = search.cv_results_
    i = search.best_index_

    n_rows = int(y_train.size)
    n_positive = int(y_train.sum())

    row = dict.fromkeys(METRIC_KEYS)
    row.update(
        {
            "kind": "model",
            "score_type": "probability",
            "split": "cv",
            "n_rows": n_rows,
            "n_positive": n_positive,
            "base_rate": n_positive / n_rows if n_rows else 0.0,
            "roc_auc": float(results["mean_test_roc_auc"][i]),
            "pr_auc": float(results["mean_test_pr_auc"][i]),
            "brier": -float(results["mean_test_brier"][i]),
            "note": f"{search.n_splits_}-fold, std {float(results['std_test_roc_auc'][i]):.4f}",
        }
    )
    return row


def evaluate_on_test(
    y_test: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Evaluates the input Pipeline using the test dataset."""
    return build_row(
        y_true=y_test,
        scores=y_proba,
        kind="model",
        split="test",
        score_type="probability",
        threshold=threshold,
        n_flagged_cap=None,
        note=None,
    )


def evaluate_on_unseen_customers(
    df_processed: pd.DataFrame,
    test_cutoff: str,
    y_test: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """Evaluates a Pipeline's probability prediction using the unseen customers in the test dataset."""
    y_test = np.asarray(y_test).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    cutoff = pd.Timestamp(test_cutoff)
    train_ids = set(df_processed.loc[df_processed["cutoff"] < cutoff, "customer_id"])
    test_ids = df_processed.loc[
        df_processed["cutoff"] == cutoff, "customer_id"
    ].to_numpy()
    unseen = ~np.isin(test_ids, list(train_ids))

    y_unseen = y_test[unseen]
    too_small = (unseen.sum() < 50) or (len(np.unique(y_unseen)) < 2)

    return build_row(
        y_true=y_test[unseen],
        scores=y_proba[unseen],
        kind="model",
        split="test_unseen",
        score_type="probability",
        threshold=None,
        n_flagged_cap=None,
        note="too few unseen customers or only one class present"
        if too_small
        else None,
    )


def build_comparison_table(
    cv_metrics: dict,
    test_metrics: dict,
    unseen_test_metrics: dict,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """Compiles all evaluation data into a single table."""
    return pd.concat(
        [
            baselines,
            pd.DataFrame(
                [cv_metrics, test_metrics, unseen_test_metrics],
                index=[
                    f"CV",
                    f"Test Dataset",
                    f"Test Dataset with Unseen Customers only",
                ],
            ),
        ]
    )[METRIC_KEYS]
