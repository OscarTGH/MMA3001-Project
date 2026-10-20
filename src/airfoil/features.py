"""Build model input matrices from the raw dataset."""

import numpy as np

# The five model inputs, in a fixed order.
FEATURE_COLUMNS = ["frequency_hz", "alpha_deg", "chord_m", "velocity_ms", "delta_star_m"]


def basic_features(df, log_transform=False):
    """Return the five raw inputs as a NumPy array.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the columns in ``FEATURE_COLUMNS``.
    log_transform : bool, optional
        If True, replace frequency and delta* with their log10, because
        both are strongly right-skewed (default False).

    Returns
    -------
    numpy.ndarray, shape (n_rows, 5)
        Columns in the order of ``FEATURE_COLUMNS``.
    """
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float).copy()
    if log_transform:
        X[:, 0] = np.log10(X[:, 0])   # frequency
        X[:, 4] = np.log10(X[:, 4])   # delta*
    return X