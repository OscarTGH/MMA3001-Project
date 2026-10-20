"""Physics-based scaling-law model for airfoil self-noise (BPM-style)."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from airfoil.numerics import solve_lstsq


class ScalingLawModel(BaseEstimator, RegressorMixin):
    """Trailing-edge noise scaling law with a fitted Strouhal spectrum shape.

    Model:
        SPL = n*10*log10(U) + m*10*log10(delta*) + g(log10 St) + c*alpha + const

    where St = f * delta* / U and g is a polynomial. With
    ``fit_exponents=False`` the exponents are fixed by theory (n = 5, m = 1);
    with ``fit_exponents=True`` they are fitted from the data.

    Expects the five inputs in the order of ``airfoil.features.FEATURE_COLUMNS``
    (frequency [Hz], alpha [deg], chord [m], velocity [m/s], delta* [m]).

    Parameters
    ----------
    degree : int, optional
        Degree of the polynomial g in log10(St) (default 3).
    fit_exponents : bool, optional
        If True, fit the velocity and delta* exponents instead of fixing
        them at 5 and 1 (default False).
    """

    def __init__(self, degree=3, fit_exponents=False):
        self.degree = degree
        self.fit_exponents = fit_exponents

    def _log_strouhal(self, X):
        """log10 of the Strouhal number f * delta* / U for each row."""
        return np.log10(X[:, 0] * X[:, 4] / X[:, 3])

    def _design(self, X):
        """Build the matrix of terms whose coefficients are fitted."""
        # Centre log St so the polynomial columns are well conditioned.
        s = self._log_strouhal(X) - self.log_st_mean_
        cols = [np.ones(len(X))]
        cols += [s**k for k in range(1, self.degree + 1)]
        cols.append(X[:, 1])                      # angle of attack
        if self.fit_exponents:
            cols.append(np.log10(X[:, 3]))        # velocity
            cols.append(np.log10(X[:, 4]))        # delta*
        return np.column_stack(cols)

    def _fixed_scaling(self, X):
        """Theory part not fitted: 50*log10(U) + 10*log10(delta*)."""
        if self.fit_exponents:
            return np.zeros(len(X))
        return 50 * np.log10(X[:, 3]) + 10 * np.log10(X[:, 4])

    def fit(self, X, y):
        """Fit the spectrum-shape (and optionally exponent) coefficients.

        Parameters
        ----------
        X : array_like, shape (n_rows, 5)
            Inputs in FEATURE_COLUMNS order, raw units.
        y : array_like, shape (n_rows,)
            Measured SPL in dB.

        Returns
        -------
        ScalingLawModel
            The fitted model (allows chaining).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.log_st_mean_ = np.mean(self._log_strouhal(X))
        # Least squares with the stable SVD solver from Phase 2.
        self.coef_ = solve_lstsq(self._design(X), y - self._fixed_scaling(X))
        return self

    def predict(self, X):
        """Predict SPL in dB.

        Parameters
        ----------
        X : array_like, shape (n_rows, 5)
            Inputs in FEATURE_COLUMNS order, raw units.

        Returns
        -------
        numpy.ndarray, shape (n_rows,)
            Predicted SPL in dB.
        """
        X = np.asarray(X, dtype=float)
        return self._design(X) @ self.coef_ + self._fixed_scaling(X)

    @property
    def velocity_exponent_(self):
        """Exponent n in SPL ~ U**n (5 if fixed by theory)."""
        if not self.fit_exponents:
            return 5.0
        return self.coef_[self.degree + 2] / 10

    @property
    def delta_star_exponent_(self):
        """Exponent m in SPL ~ delta***m (1 if fixed by theory)."""
        if not self.fit_exponents:
            return 1.0
        return self.coef_[self.degree + 3] / 10