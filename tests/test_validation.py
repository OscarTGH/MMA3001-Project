"""Tests for airfoil.validation."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from airfoil.data import load_data
from airfoil.validation import (
    cross_validate_model,
    make_cv_schemes,
    regression_metrics,
    setup_groups,
)


def test_metrics_perfect_prediction():
    """A perfect prediction has zero error and R squared of 1."""
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["rmse"] == pytest.approx(0)
    assert m["mae"] == pytest.approx(0)
    assert m["max_error"] == pytest.approx(0)
    assert m["r2"] == pytest.approx(1)


def test_metrics_hand_calculated():
    """Predicting the mean (2) for [1, 2, 3]: errors are 1, 0, -1.

    RMSE = sqrt(2/3), MAE = 2/3, max = 1, and R squared = 0 because
    guessing the average explains none of the variation.
    """
    m = regression_metrics([1, 2, 3], [2, 2, 2])
    assert m["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["max_error"] == pytest.approx(1)
    assert m["r2"] == pytest.approx(0)


def test_setup_groups_counts_106_setups():
    """The dataset contains 106 distinct wind-tunnel set-ups."""
    groups = setup_groups(load_data())
    assert len(np.unique(groups)) == 106


def test_grouped_cv_keeps_setups_together():
    """No set-up ever appears in both the training and test data of a fold."""
    df = load_data()
    cv, groups = make_cv_schemes(df)["grouped"]
    for train_idx, test_idx in cv.split(df, groups=groups):
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])


def test_leave_one_chord_out_has_six_folds():
    """There is one fold per chord length (6 chords)."""
    df = load_data()
    cv, groups = make_cv_schemes(df)["leave_one_chord_out"]
    assert cv.get_n_splits(groups=groups) == 6


def test_cross_validate_model_on_exact_line():
    """Linear regression on noise-free linear data is perfect on every fold."""
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 3 * X.ravel() + 2
    results = cross_validate_model(LinearRegression(), X, y,
                                   KFold(5, shuffle=True, random_state=0))
    assert len(results) == 5
    assert results["rmse"].max() == pytest.approx(0, abs=1e-10)


from airfoil.validation import (
    cross_val_predictions,
    cross_validate_model,
    make_cv_schemes,
    nested_cv,
    regression_metrics,
    setup_groups,
)

def test_nested_cv_returns_one_row_per_outer_fold():
    """Nested CV gives one result per outer fold, with params from the grid."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.1, size=120)
    groups = np.repeat(np.arange(30), 4)   # 30 groups of 4 rows

    results = nested_cv(Ridge(), {"alpha": [0.01, 1.0]}, X, y, groups,
                        n_outer=3, n_inner=3)
    assert len(results) == 3
    assert all(p["alpha"] in (0.01, 1.0) for p in results["best_params"])
    assert results["rmse"].max() < 0.5

def test_cross_val_predictions_fills_every_row():
    """Every row gets a prediction, and an exact line is predicted exactly."""
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 3 * X.ravel() + 2
    y_pred = cross_val_predictions(LinearRegression(), X, y,
                                   KFold(5, shuffle=True, random_state=0))
    assert not np.isnan(y_pred).any()
    assert y_pred == pytest.approx(y, abs=1e-10)