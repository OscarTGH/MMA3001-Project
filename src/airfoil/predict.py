"""Easy-to-use prediction interface for the final airfoil self-noise model."""

import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from airfoil.features import FEATURE_COLUMNS, basic_features
from airfoil.numerics import db_sum

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "final_model.joblib"

# Ranges covered by the NASA wind-tunnel data (min, max).
TESTED_RANGES = {
    "frequency_hz": (200.0, 20000.0),
    "alpha_deg": (0.0, 22.2),
    "chord_m": (0.0254, 0.3048),
    "velocity_ms": (31.7, 71.3),
    "delta_star_m": (0.000400682, 0.0584113),
}

# The 21 third-octave band centres in the dataset (Hz).
BAND_CENTRES_HZ = np.array([200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
                            2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000],
                           dtype=float)


class ExtrapolationWarning(UserWarning):
    """An input lies outside the range covered by the training data."""


class UnreliableRegionWarning(UserWarning):
    """Inputs fall where validation showed large errors (small chord, high angle)."""


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    """Load the saved final model (cached, so the file is only read once).

    Parameters
    ----------
    path : pathlib.Path, optional
        Location of the saved model (default ``models/final_model.joblib``).

    Returns
    -------
    scikit-learn estimator
        The trained MLP pipeline, expecting log-transformed basic features.
    """
    return joblib.load(path)


def _as_frame(frequency_hz, alpha_deg, chord_m, velocity_ms, delta_star_m):
    """Broadcast the five inputs together and return them as a DataFrame."""
    try:
        values = [np.asarray(v, dtype=float)
                  for v in (frequency_hz, alpha_deg, chord_m, velocity_ms, delta_star_m)]
    except (TypeError, ValueError) as err:
        raise ValueError(f"All inputs must be numbers: {err}") from None
    arrays = np.broadcast_arrays(*values)
    frame = pd.DataFrame({name: np.ravel(a) for name, a in zip(FEATURE_COLUMNS, arrays)})
    return frame, arrays[0].shape


def check_inputs(df):
    """Reject invalid inputs and warn about risky ones.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per prediction, with the columns in ``FEATURE_COLUMNS``.

    Raises
    ------
    ValueError
        If any value is missing, NaN or infinite, or if frequency, chord,
        velocity or delta* is not positive.

    Warns
    -----
    ExtrapolationWarning
        If any input is outside ``TESTED_RANGES``.
    UnreliableRegionWarning
        If any row has chord <= 0.0508 m and angle of attack > 15 degrees.
    """
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Inputs must not be missing, NaN or infinite.")
    for col in ["frequency_hz", "chord_m", "velocity_ms", "delta_star_m"]:
        if (df[col] <= 0).any():
            raise ValueError(f"{col} must be positive.")

    outside = [col for col, (lo, hi) in TESTED_RANGES.items()
               if ((df[col] < lo) | (df[col] > hi)).any()]
    if outside:
        warnings.warn(f"Inputs outside the tested range: {', '.join(outside)}. "
                      "Predictions there could not be validated.",
                      ExtrapolationWarning, stacklevel=3)

    if ((df["chord_m"] <= 0.0508) & (df["alpha_deg"] > 15)).any():
        warnings.warn("Small chord with high angle of attack: validation errors reached "
                      "about 20 dB in this region.",
                      UnreliableRegionWarning, stacklevel=3)


def predict_spl(frequency_hz, alpha_deg, chord_m, velocity_ms, delta_star_m):
    """Predict the scaled sound pressure level of a NACA 0012 blade section.

    Inputs can be single numbers or arrays; they are broadcast together
    (e.g. an array of frequencies with single values for the rest).

    Parameters
    ----------
    frequency_hz : float or array_like
        Third-octave band centre frequency [Hz]. Tested range 200-20,000.
    alpha_deg : float or array_like
        Angle of attack [degrees]. Tested range 0-22.2.
    chord_m : float or array_like
        Chord length [m]. Tested range 0.0254-0.3048.
    velocity_ms : float or array_like
        Free-stream velocity [m/s]. Tested range 31.7-71.3.
    delta_star_m : float or array_like
        Suction-side displacement thickness [m]. Tested range 0.0004-0.058.
        Must be estimated consistently with chord, velocity and angle, as in
        the original NASA data.

    Returns
    -------
    float or numpy.ndarray
        Predicted scaled SPL in dB (a float if all inputs were single numbers).

    Raises
    ------
    ValueError
        For missing, non-numeric, infinite or non-physical inputs.

    Warns
    -----
    ExtrapolationWarning, UnreliableRegionWarning
        See ``check_inputs``.
    """
    df, shape = _as_frame(frequency_hz, alpha_deg, chord_m, velocity_ms, delta_star_m)
    check_inputs(df)
    pred = load_model().predict(basic_features(df, log_transform=True)).reshape(shape)
    return float(pred) if pred.ndim == 0 else pred


def predict_spectrum(alpha_deg, chord_m, velocity_ms, delta_star_m,
                     frequencies_hz=BAND_CENTRES_HZ):
    """Predicted spectrum (SPL at each frequency) for one operating condition.

    Parameters
    ----------
    alpha_deg, chord_m, velocity_ms, delta_star_m : float
        Operating condition (see ``predict_spl`` for units and ranges).
    frequencies_hz : array_like, optional
        Frequencies to predict at (default: the 21 third-octave band centres).

    Returns
    -------
    pandas.DataFrame
        Columns ``frequency_hz`` and ``spl_db``.
    """
    freqs = np.asarray(frequencies_hz, dtype=float)
    spl = predict_spl(freqs, alpha_deg, chord_m, velocity_ms, delta_star_m)
    return pd.DataFrame({"frequency_hz": freqs, "spl_db": np.atleast_1d(spl)})


def overall_level(alpha_deg, chord_m, velocity_ms, delta_star_m):
    """Overall level (dB): energy sum of the predicted 21-band spectrum.

    Phase 6 showed this band sum is within 0.02 dB of a converged integral
    of the model's continuous spectrum.

    Parameters
    ----------
    alpha_deg, chord_m, velocity_ms, delta_star_m : float
        Operating condition (see ``predict_spl`` for units and ranges).

    Returns
    -------
    float
        Overall level in dB.
    """
    spectrum = predict_spectrum(alpha_deg, chord_m, velocity_ms, delta_star_m)
    return db_sum(spectrum["spl_db"])