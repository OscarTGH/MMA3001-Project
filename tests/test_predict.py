"""Tests for airfoil.predict."""

import warnings

import numpy as np
import pytest

from airfoil.data import load_data
from airfoil.features import basic_features
from airfoil.numerics import db_sum
from airfoil.predict import (
    ExtrapolationWarning,
    UnreliableRegionWarning,
    load_model,
    overall_level,
    predict_spectrum,
    predict_spl,
)

# A tested condition: largest chord, 0 degrees, 55.5 m/s
GOOD = dict(alpha_deg=0.0, chord_m=0.3048, velocity_ms=55.5, delta_star_m=0.00283)


def test_matches_model_directly():
    """predict_spl gives the same answer as calling the saved model directly."""
    df = load_data().iloc[[0]]
    direct = load_model().predict(basic_features(df, log_transform=True))[0]
    row = df.iloc[0]
    assert predict_spl(row["frequency_hz"], row["alpha_deg"], row["chord_m"],
                       row["velocity_ms"], row["delta_star_m"]) == pytest.approx(direct)


def test_in_range_input_gives_no_warning_and_plausible_value():
    """A tested condition predicts a plausible level without any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # turn any warning into a failure
        spl = predict_spl(frequency_hz=2000, **GOOD)
    assert 100 < spl < 145


def test_array_of_frequencies_broadcasts():
    """An array of frequencies with single values for the rest gives one SPL each."""
    spl = predict_spl(np.array([500, 1000, 2000]), **GOOD)
    assert spl.shape == (3,)


def test_missing_value_raises():
    with pytest.raises(ValueError):
        predict_spl(frequency_hz=np.nan, **GOOD)


def test_non_physical_value_raises():
    with pytest.raises(ValueError):
        predict_spl(frequency_hz=-100, **GOOD)


def test_outside_tested_range_warns():
    """A speed of 100 m/s is beyond the tested 31.7-71.3 m/s."""
    with pytest.warns(ExtrapolationWarning):
        predict_spl(frequency_hz=2000, alpha_deg=0.0, chord_m=0.3048,
                    velocity_ms=100, delta_star_m=0.00283)


def test_small_chord_high_angle_warns():
    """Validation found errors up to about 20 dB for small chords at high angle."""
    with pytest.warns(UnreliableRegionWarning):
        predict_spl(frequency_hz=1250, alpha_deg=20.0, chord_m=0.0254,
                    velocity_ms=71.3, delta_star_m=0.0214)


def test_spectrum_and_overall_level_agree():
    """The overall level is the energy sum of the 21-band spectrum."""
    spectrum = predict_spectrum(**GOOD)
    assert len(spectrum) == 21
    assert overall_level(**GOOD) == pytest.approx(db_sum(spectrum["spl_db"]))