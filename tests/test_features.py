"""Tests for airfoil.features."""

import numpy as np

from airfoil.data import load_data
from airfoil.features import FEATURE_COLUMNS, basic_features


def test_basic_features_shape():
    """Five input columns for all 1503 rows."""
    assert basic_features(load_data()).shape == (1503, 5)


def test_log_transform_changes_only_frequency_and_delta_star():
    """Log mode takes log10 of frequency and delta*, leaving the rest unchanged."""
    df = load_data()
    raw = basic_features(df)
    logged = basic_features(df, log_transform=True)
    assert np.allclose(logged[:, 0], np.log10(raw[:, 0]))
    assert np.allclose(logged[:, 4], np.log10(raw[:, 4]))
    assert np.allclose(logged[:, 1:4], raw[:, 1:4])


def test_basic_features_does_not_modify_dataframe():
    """Building features must not change the original data."""
    df = load_data()
    before = df[FEATURE_COLUMNS].copy()
    basic_features(df, log_transform=True)
    assert df[FEATURE_COLUMNS].equals(before)