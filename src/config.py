"""Configures constant variables from config.yaml file to current file."""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataCfg:
    raw_path: Path
    interim_path: Path
    processed_path: Path

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, PROJECT_ROOT / getattr(self, f.name))


@dataclass(frozen=True)
class PrepCfg:
    column_map: dict
    non_products: list[str]
    purge_report_persist_path: Path

    def __post_init__(self):
        for path in ["purge_report_persist_path"]:
            object.__setattr__(self, path, PROJECT_ROOT / getattr(self, path))


@dataclass(frozen=True)
class LabellingCfg:
    cutoffs: list[str]
    churn_horizon: int
    eligibility_lookback: int
    lookback_days: list[int]


@dataclass(frozen=True)
class ModelCfg:
    random_state: int
    test_cutoff: str
    heuristic_baseline: int
    model_persist_path: Path
    metadata_persist_path: Path
    evaluation_report_persist_path: Path
    curve_data_persist_path: Path
    evaluation_arrays_persist_path: Path

    def __post_init__(self):
        for path in (
            "model_persist_path",
            "metadata_persist_path",
            "evaluation_report_persist_path",
            "curve_data_persist_path",
            "evaluation_arrays_persist_path",
        ):
            object.__setattr__(self, path, PROJECT_ROOT / getattr(self, path))


@dataclass(frozen=True)
class EconomicsCfg:
    cost_per_contact: float
    average_customer_lifetime_value: float
    save_rate: float
    sensitivity_spread: float
    sensitivity_points: int
    campaign_capacity: int


@dataclass(frozen=True)
class PredictionCfg:
    medium_risk_band_multiplier: float


@dataclass(frozen=True)
class DatabaseCfg:
    db_path: Path

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, PROJECT_ROOT / getattr(self, f.name))


@dataclass(frozen=True)
class Config:
    data: DataCfg
    prep: PrepCfg
    labelling: LabellingCfg
    model: ModelCfg
    economics: EconomicsCfg
    prediction: PredictionCfg
    database: DatabaseCfg


def load_config() -> Config:
    """Unpacks constant variables from config.yaml into a Config class object."""
    path = PROJECT_ROOT / "config.yaml"
    if not path.exists():
        raise RuntimeError(f"config.yaml could not be found under {PROJECT_ROOT}")
    with open(path) as yamlfile:
        raw = yaml.safe_load(yamlfile)
    return Config(
        data=DataCfg(**raw["data"]),
        prep=PrepCfg(**raw["prep"]),
        labelling=LabellingCfg(**raw["labelling"]),
        model=ModelCfg(**raw["model"]),
        economics=EconomicsCfg(**raw["economics"]),
        prediction=PredictionCfg(**raw["prediction"]),
        database=DatabaseCfg(**raw["database"]),
    )
