"""Unit tests for the src.predict module."""

import numpy as np

from src.predict import assign_risk_band


def test_assign_risk_band():
    test = assign_risk_band(
        churn_proba=np.array([0.2, 0.5, 0.8]),
        metadata={"chosen_threshold": 0.6},
        medium_risk_band_multiplier=0.5,
    )
    assert test[0] == "Low"
    assert test[1] == "Medium"
    assert test[2] == "High"
