"""
Engine for one command repository set-up: a main() function for the src package.
Rebuilds model using default config values and writes predictions to SQLite database.
"""

from src.config import load_config
from src.data_prep import prepare_data
from src.features import build_dataset
from src.labelling import label_customers
from src.train import generate_model
from src.predict import score_all


def main() -> None:
    """Rebuilds model using default config values and writes predictions to SQLite database."""

    print(
        "\n Running Pipeline for Retain Churn Prediction Model construction and repository set-up."
    )

    # Load default config values
    config = load_config()

    # Cleans raw dataset (raw -> interim data) for feature engineering
    prepare_data(
        raw_path=config.data.raw_path,
        column_map=config.prep.column_map,
        non_products=config.prep.non_products,
        interim_path=config.data.interim_path,
        purge_report_path=config.prep.purge_report_persist_path,
    )

    # Builds features using cleaned dataset (interim -> processed data)
    build_dataset(
        interim_path=config.data.interim_path,
        cutoffs=config.labelling.cutoffs,
        churn_horizon=config.labelling.churn_horizon,
        eligibility_lookback=config.labelling.eligibility_lookback,
        lookback_days=config.labelling.lookback_days,
        processed_path=config.data.processed_path,
    )

    # Adds 'churned' label to processed dataset
    label_customers(
        interim_path=config.data.interim_path,
        cutoffs=config.labelling.cutoffs,
        churn_horizon=config.labelling.churn_horizon,
        eligibility_lookback=config.labelling.eligibility_lookback,
        processed_path=config.data.processed_path,
    )

    # Builds, trains, tunes, fits, evaluates and persists model
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

    # Predicts churn probability at the latest transaction date and writes predictions to SQLite database
    score_all(
        model_persist_path=config.model.model_persist_path,
        metadata_persist_path=config.model.metadata_persist_path,
        interim_path=config.data.interim_path,
        eligibility_lookback=config.labelling.eligibility_lookback,
        lookback_days=config.labelling.lookback_days,
        medium_risk_band_multiplier=config.prediction.medium_risk_band_multiplier,
        db_path=config.database.db_path,
    )

    print("\n Pipeline complete. Run `flask --app app run` to view the dashboard.\n")


if __name__ == "__main__":
    main()
