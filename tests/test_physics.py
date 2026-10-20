"""Tests for airfoil.physics.ScalingLawModel."""

import numpy as np
import pytest

from airfoil.data import load_data
from airfoil.features import basic_features
from airfoil.physics import ScalingLawModel
from airfoil.validation import cross_validate_model, make_cv_schemes


def _synthetic_inputs(n=400, seed=0):
    """Random inputs spanning roughly the dataset's ranges."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(200, 20000, n),    # frequency
        rng.uniform(0, 20, n),         # alpha
        rng.uniform(0.02, 0.3, n),     # chord
        rng.uniform(30, 72, n),        # velocity
        rng.uniform(1e-3, 5e-2, n),    # delta*
    ])


def _synthetic_spl(X, velocity_exponent):
    """Noise-free SPL following a known scaling law."""
    log_st = np.log10(X[:, 0] * X[:, 4] / X[:, 3])
    return (10 * velocity_exponent * np.log10(X[:, 3]) + 10 * np.log10(X[:, 4])
            + 120 - 3 * log_st - 0.8 * log_st**2 + 0.1 * X[:, 1])


def test_fixed_model_fits_exact_u5_law():
    """Data that truly follows U^5 is fitted perfectly by the fixed model."""
    X = _synthetic_inputs()
    y = _synthetic_spl(X, velocity_exponent=5)
    model = ScalingLawModel(degree=3).fit(X, y)
    assert np.max(np.abs(model.predict(X) - y)) == pytest.approx(0, abs=1e-8)


def test_fitted_exponent_recovers_known_value():
    """If the data follows U^4, the fitted velocity exponent comes out as 4."""
    X = _synthetic_inputs()
    y = _synthetic_spl(X, velocity_exponent=4)
    model = ScalingLawModel(degree=3, fit_exponents=True).fit(X, y)
    assert model.velocity_exponent_ == pytest.approx(4, abs=1e-6)
    assert model.delta_star_exponent_ == pytest.approx(1, abs=1e-6)


def test_works_with_cross_validation_on_real_data():
    """The model plugs into cross_validate_model and gives finite errors."""
    df = load_data()
    X = basic_features(df)
    y = df["spl_db"].to_numpy()
    cv, groups = make_cv_schemes(df)["grouped"]
    results = cross_validate_model(ScalingLawModel(), X, y, cv, groups)
    assert len(results) == 5
    assert np.isfinite(results["rmse"]).all()