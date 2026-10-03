"""
Builds, trains, and tunes hyperparameters for a churn prediction Pipeline.
Persists generated metadata, an evaluation report and Chart.js curve data alongside the
fitted/tuned Pipeline in the 'models' directory.
"""

from datetime import datetime
import joblib
import json
from pathlib import Path

import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    BaseCrossValidator,
    cross_validate,
    GridSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import EconomicsCfg, load_config
from src.evaluate import (
    evaluate_baselines,
    generate_cv_metrics,
    evaluate_on_test,
    evaluate_on_unseen_customers,
    build_comparison_table,
)
from src.economics import select_threshold
from src.curve import generate_curve_data


def temporal_split(
    df_processed: pd.DataFrame, test_cutoff: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, dict[str, int]]:
    """
    Splits the processed dataset into X and y train and test sets (with a 'groups' identifier series for the train set)
    based on the test cutoff date.
    """
    df = df_processed.copy()
    cutoff = pd.Timestamp(test_cutoff)

    # Separates train and test sets by test cutoff date
    df_train = df[df["cutoff"] < cutoff]
    df_test = df[df["cutoff"] == cutoff]

    # Splits train set into X_train, y_train and groups
    X_train = df_train.drop(
        columns=["customer_id", "cutoff", "churn_horizon", "churned"]
    )
    y_train = df_train["churned"]
    groups = df_train["customer_id"]

    # Splits test set into X_test and y_test
    X_test = df_test.drop(columns=["customer_id", "cutoff", "churn_horizon", "churned"])
    y_test = df_test["churned"]

    # Generates split report
    overlap = set(df_train["customer_id"]) & set(df_test["customer_id"])
    report = {
        "df_train_rows": len(df_train),
        "df_train_customers": df_train["customer_id"].nunique(),
        "df_test_rows": len(df_test),
        "df_test_customers": df_test["customer_id"].nunique(),
        "overlapping_customers": len(overlap),
    }

    return X_train, y_train, groups, X_test, y_test, report


def build_pipeline(lookback_days: list[int], estimator=None) -> Pipeline:
    """Generates the preprocessor and estimator Pipeline."""
    if estimator is None:
        estimator = LogisticRegression(max_iter=5000)

    numerical = [
        "recency",
        "tenure",
        f"spend_momentum_{lookback_days[0]}d_{lookback_days[1]}d",
        f"spend_momentum_{lookback_days[1]}d_{lookback_days[2]}d",
        f"order_momentum_{lookback_days[0]}d_{lookback_days[1]}d",
        f"order_momentum_{lookback_days[1]}d_{lookback_days[2]}d",
        "return_rate",
    ]

    numerical_log = [
        "frequency",
        "monetary",
        "mean_order_value",
        "avg_inter_purchase_gap",
        "std_inter_purchase_gap",
        "recency_ratio",
        f"spend_{lookback_days[0]}d",
        f"spend_{lookback_days[1]}d",
        f"spend_{lookback_days[2]}d",
        f"orders_{lookback_days[0]}d",
        f"orders_{lookback_days[1]}d",
        f"orders_{lookback_days[2]}d",
        "distinct_products",
        "mean_order_quantity",
    ]

    categorical = ["country"]

    numerical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )

    numerical_log_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipe, numerical),
            ("numerical_log", numerical_log_pipe, numerical_log),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def build_candidates(
    lookback_days: list[int], random_state: int
) -> dict[str, Pipeline]:
    """Generates a dictionary of unfitted Pipelines."""
    return {
        "logistic": build_pipeline(
            lookback_days, LogisticRegression(max_iter=5000, random_state=random_state)
        ),
        "random_forest": build_pipeline(
            lookback_days,
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "gradient_boosting": build_pipeline(
            lookback_days,
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=400,
                min_samples_leaf=20,
                early_stopping=False,
                n_iter_no_change=20,
                random_state=random_state,
            ),
        ),
    }


def compare_candidates(
    candidates: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    cv: BaseCrossValidator,
) -> pd.DataFrame:
    """
    Runs group-aware cross-validation on each candidate at default hyperparameters and returns a
    table of the mean and standard deviation.
    """
    rows = []
    for name, candidate in candidates.items():
        cv_results = cross_validate(
            candidate,
            X_train,
            y_train,
            groups=groups,
            cv=cv,
            scoring=["roc_auc", "average_precision"],
            return_train_score=True,
        )
        rows.append(
            {
                "model": name,
                "roc_auc": cv_results["test_roc_auc"].mean(),
                "roc_auc_std": cv_results["test_roc_auc"].std(),
                "pr_auc": cv_results["test_average_precision"].mean(),
                "pr_auc_std": cv_results["test_average_precision"].std(),
                "pr_auc_base_rate": float(y_train.mean()),
                "train_roc_auc": cv_results["train_roc_auc"].mean(),
                "fit_seconds": cv_results["fit_time"].mean(),
            }
        )

    results = (
        pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
    )

    return results


def tune_winner(
    winner_name: str,
    lookback_days: list[int],
    random_state: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    cv: BaseCrossValidator,
) -> GridSearchCV:
    """Tunes the hyperparameters for the winning Pipeline using GridSearchCV."""
    winner = build_candidates(lookback_days, random_state)[winner_name]

    param_grids = {
        "logistic": {
            "classifier__C": np.logspace(-3, 2, 12),
            "classifier__l1_ratio": [0.0, 0.5, 1.0],
            "classifier__solver": ["saga"],
        },
        "random_forest": {
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_leaf": [1, 5, 10, 20],
            "classifier__max_features": ["sqrt", "log2", 0.5],
        },
        "gradient_boosting": {
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "classifier__max_leaf_nodes": [15, 31, 63],
            "classifier__min_samples_leaf": [10, 20, 40],
            "classifier__l2_regularization": [0.0, 0.1, 1.0],
        },
    }
    param_grid = param_grids.get(winner_name)
    if param_grid is None:
        raise ValueError(f"No parameter grid defined for '{winner_name}'")

    search = GridSearchCV(
        estimator=winner,
        param_grid=param_grid,
        scoring={
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
            "brier": "neg_brier_score",
        },
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=groups)

    return search


def persist(
    model: Pipeline,
    metadata: dict,
    evaluation_report: dict,
    curve_data: dict,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_persist_path: Path,
    metadata_persist_path: Path,
    evaluation_report_persist_path: Path,
    curve_data_persist_path: Path,
    evaluation_arrays_persist_path: Path,
) -> None:
    """Persists the input Pipeline and metadata/evaluation reports."""
    for path in (
        model_persist_path,
        metadata_persist_path,
        evaluation_report_persist_path,
        curve_data_persist_path,
        evaluation_arrays_persist_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_persist_path)

    paths = (
        metadata_persist_path,
        evaluation_report_persist_path,
        curve_data_persist_path,
    )
    json_dicts = (metadata, evaluation_report, curve_data)

    for path, json_dict in zip(paths, json_dicts):
        path.write_text(json.dumps(json_dict, indent=2, default=str))

    np.savez_compressed(
        evaluation_arrays_persist_path,
        y_true=y_true,
        y_proba=y_proba,
    )


def generate_model(
    processed_path: Path,
    random_state: int,
    test_cutoff: str,
    economics: EconomicsCfg,
    heuristic_baseline: int,
    cutoffs: list[str],
    churn_horizon: int,
    eligibility_lookback: int,
    lookback_days: list[int],
    model_persist_path: Path,
    metadata_persist_path: Path,
    evaluation_report_persist_path: Path,
    curve_data_persist_path: Path,
    evaluation_arrays_persist_path: Path,
) -> None:
    """Engine for building, fitting and evaluating a Pipeline with hyperparameter tuning."""

    print("\n ----------------- train.py start -----------------")

    print("\n (1) Loading processed dataset into train.py")
    df_processed = pd.read_parquet(processed_path)
    df_processed["cutoff"] = pd.to_datetime(df_processed["cutoff"])

    print("\n (2) Splitting dataset into training and testing datasets")
    X_train, y_train, groups, X_test, y_test, split_report = temporal_split(
        df_processed, test_cutoff
    )
    print("\n", split_report)

    print("\n (3) Building model candidates")
    candidates = build_candidates(lookback_days, random_state)

    print("\n (4) Comparing model candidates via cross-validation")
    cv_strategy = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    comparison = compare_candidates(
        candidates=candidates,
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        cv=cv_strategy,
    )
    winner_name = comparison.index[0]
    print("\n", comparison)

    print("\n (5) Tuning winning model hyperparameters using GridSearchCV")
    search = tune_winner(
        winner_name=winner_name,
        lookback_days=lookback_days,
        random_state=random_state,
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        cv=cv_strategy,
    )
    best_estimator = search.best_estimator_

    n_scored = int(y_test.size)
    y_true = y_test.to_numpy()
    y_proba = best_estimator.predict_proba(X_test)[:, 1]

    print("\n (6) Selecting a contact threshold for a retention campaign")
    threshold = select_threshold(
        y_true=y_test.to_numpy(),
        y_proba=y_proba,
        n_scored=n_scored,
        cost=economics.cost_per_contact,
        customer_lifetime_value=economics.average_customer_lifetime_value,
        save_rate=economics.save_rate,
        sensitivity_spread=economics.sensitivity_spread,
        sensitivity_points=economics.sensitivity_points,
        capacity=economics.campaign_capacity,
    )
    print("\n", threshold)

    print("\n (7) Evaluating model baselines")
    baselines = evaluate_baselines(
        X_test=X_test,
        y_test=y_true,
        capacity=economics.campaign_capacity,
        random_state=random_state,
        heuristic_baseline=heuristic_baseline,
        n_scored=n_scored,
    )
    print("\n", baselines)

    print("\n (8) Generating model cross-validation metrics")
    cv_metrics = generate_cv_metrics(search=search, y_train=y_train)
    print("\n", cv_metrics)

    print("\n (9) Evaluating tuned model using testing datasets")
    test_metrics = evaluate_on_test(
        y_test=y_true, y_proba=y_proba, threshold=threshold["chosen_threshold"]
    )
    print("\n", test_metrics)

    print(
        "\n (10) Evaluating tuned model using only unseen customers from testing dataset"
    )
    unseen_metrics = evaluate_on_unseen_customers(
        df_processed=df_processed,
        test_cutoff=test_cutoff,
        y_test=y_true,
        y_proba=y_proba,
    )
    print("\n", unseen_metrics)

    print("\n (11) Generating curve data for Chart.js frontend")
    curve = generate_curve_data(y_true=y_true, y_proba=y_proba)

    print("\n (12) Building model metadata and evaluation reports")
    timestamp = datetime.now().isoformat()

    map = {
        "logistic": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }

    metadata = {
        "trained_at": timestamp,
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        "training_cutoffs": [
            cutoff
            for cutoff in cutoffs
            if pd.Timestamp(cutoff) < pd.Timestamp(test_cutoff)
        ],
        "test_cutoff": test_cutoff,
        "churn_horizon": churn_horizon,
        "eligibility_lookback": eligibility_lookback,
        "feature_columns": list(X_train.columns),
        "winner": map[winner_name],
        "best_params": search.best_params_,
        "chosen_threshold": threshold["chosen_threshold"],
        "cv_roc_auc": float(search.best_score_),
        "test_roc_auc": float(test_metrics["roc_auc"]),
    }

    evaluation_table = build_comparison_table(
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        unseen_test_metrics=unseen_metrics,
        baselines=baselines,
    )

    evaluation_report = {
        "trained_at": timestamp,
        "split_report": split_report,
        "candidate_comparison": comparison.to_dict(orient="index"),
        "evaluation_table": json.loads(evaluation_table.to_json(orient="index")),
        "threshold": threshold,
    }

    curve_data = {
        "trained_at": timestamp,
        **curve,
    }

    print(f"\n (13) Persisting best estimator (tuned) as {model_persist_path}")
    print(f"\n (14) Persisting model metadata as {metadata_persist_path}")
    print(f"\n (15) Persisting evaluation report as {evaluation_report_persist_path}")
    print(f"\n (16) Persisting curve data as {curve_data_persist_path}")
    print(
        f"\n (17) Persisting y_true (y_test) and y_proba as {evaluation_arrays_persist_path}"
    )
    persist(
        model=best_estimator,
        metadata=metadata,
        evaluation_report=evaluation_report,
        curve_data=curve_data,
        y_true=y_true,
        y_proba=y_proba,
        model_persist_path=model_persist_path,
        metadata_persist_path=metadata_persist_path,
        evaluation_report_persist_path=evaluation_report_persist_path,
        curve_data_persist_path=curve_data_persist_path,
        evaluation_arrays_persist_path=evaluation_arrays_persist_path,
    )

    print("\n ------------------ train.py end ------------------")


def main() -> None:
    """Runs generate_model() with default config values."""
    config = load_config()
    generate_model(
        processed_path=config.data.processed_path,
        random_state=config.model.random_state,
        test_cutoff=config.model.test_cutoff,
        economics=config.economics,
        heuristic_baseline=config.model.heuristic_baseline,
        cutoffs=config.labelling.cutoffs,
        churn_horizon=config.labelling.churn_horizon,
        eligibility_lookback=config.labelling.eligibility_lookback,
        lookback_days=config.labelling.lookback_days,
        model_persist_path=config.model.model_persist_path,
        metadata_persist_path=config.model.metadata_persist_path,
        evaluation_report_persist_path=config.model.evaluation_report_persist_path,
        curve_data_persist_path=config.model.curve_data_persist_path,
        evaluation_arrays_persist_path=config.model.evaluation_arrays_persist_path,
    )


if __name__ == "__main__":
    main()
