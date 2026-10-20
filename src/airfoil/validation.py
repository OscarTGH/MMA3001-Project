"""Validation tools: set-up groups, error metrics and cross-validation."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

# Columns that together identify one wind-tunnel test (a "set-up").
SETUP_COLUMNS = ["chord_m", "velocity_ms", "alpha_deg"]


def setup_groups(df):
    """Label each row with the integer ID of its wind-tunnel set-up.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the columns in ``SETUP_COLUMNS``.

    Returns
    -------
    numpy.ndarray of int, shape (n_rows,)
        Set-up ID per row (0 to n_setups - 1). Rows from the same test
        share an ID.
    """
    return df.groupby(SETUP_COLUMNS).ngroup().to_numpy()


def regression_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R squared and maximum absolute error.

    Parameters
    ----------
    y_true : array_like
        Measured values (dB).
    y_pred : array_like
        Predicted values (dB).

    Returns
    -------
    dict
        Keys ``rmse``, ``mae``, ``r2`` and ``max_error``. Errors are in the
        same units as the inputs; ``r2`` is dimensionless.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred

    ss_res = np.sum(errors**2)                        # unexplained variation
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)    # total variation

    return {
        "rmse": np.sqrt(np.mean(errors**2)),
        "mae": np.mean(np.abs(errors)),
        "r2": 1 - ss_res / ss_tot,
        "max_error": np.max(np.abs(errors)),
    }


def make_cv_schemes(df, n_splits=5, seed=0):
    """Build the three cross-validation schemes used in this project.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset.
    n_splits : int, optional
        Number of folds for the random and grouped schemes (default 5).
    seed : int, optional
        Random seed for the random scheme, for repeatability (default 0).

    Returns
    -------
    dict
        Maps scheme name to ``(splitter, groups)``:

        - ``"random"``: shuffled K-fold over rows (optimistic reference).
        - ``"grouped"``: K-fold keeping whole set-ups together (primary).
        - ``"leave_one_chord_out"``: each chord length held out once.
    """
    return {
        "random": (KFold(n_splits, shuffle=True, random_state=seed), None),
        "grouped": (GroupKFold(n_splits), setup_groups(df)),
        "leave_one_chord_out": (LeaveOneGroupOut(), df["chord_m"].to_numpy()),
    }


def cross_validate_model(model, X, y, cv, groups=None):
    """Train and test a model on each fold and record its errors.

    A fresh copy of the model is trained for every fold, so no fold can
    see another fold's test data.

    Parameters
    ----------
    model : scikit-learn estimator
        Any object with ``fit`` and ``predict`` (a Pipeline is fine).
    X : numpy.ndarray, shape (n_rows, n_features)
        Input features.
    y : numpy.ndarray, shape (n_rows,)
        Target values (SPL in dB).
    cv : scikit-learn splitter
        From ``make_cv_schemes``.
    groups : numpy.ndarray or None, optional
        Group label per row, required by grouped schemes.

    Returns
    -------
    pandas.DataFrame
        One row per fold with ``fold``, ``n_test`` and the four metrics.
    """
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        fold_model = clone(model)                 # fresh, untrained copy
        fold_model.fit(X[train_idx], y[train_idx])
        y_pred = fold_model.predict(X[test_idx])
        rows.append({"fold": fold, "n_test": len(test_idx),
                     **regression_metrics(y[test_idx], y_pred)})
    return pd.DataFrame(rows)