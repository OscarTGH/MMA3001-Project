"""Numerical routines: dB combination and finite-difference derivatives."""

import numpy as np


def db_sum(levels_db):
    """Combine sound levels in dB by summing their acoustic energy.

    Decibels are logarithmic, so they cannot be added directly. Each level
    is converted to relative energy 10**(L/10), summed, and converted back.

    Parameters
    ----------
    levels_db : array_like
        Sound pressure levels in dB.

    Returns
    -------
    float
        Combined level in dB.

    Raises
    ------
    ValueError
        If ``levels_db`` is empty.

    Notes
    -----
    Computing 10^(L/10) directly can overflow, because numbers above about 10³⁰⁸ become infinity on a computer. 
    Subtracting the largest level first keeps every exponent at or below zero. 
    The answer is mathematically identical, but numerically safe. This is your Numerical Errors topic.
    L_total = L_max + 10*log10( sum 10**((L_i - L_max)/10) ).
    """
    # np.asarray convert level_db value to a float NumPy array so maths applies element-wise
    # dtype=float forces the numbers to be decimals as NumPy refuses to raise integers to negative powers.
    levels = np.asarray(levels_db, dtype = float)
    if levels.size == 0:
        raise ValueError("levels_db must contain at least one value")

    # Factor out the maximum for numerical stability (avoids overflow).
    l_max = levels.max()
    return l_max + 10 * np.log10(np.sum(10 ** ((levels - l_max) / 10)))

def forward_difference(f, x, h):
    """Forward-difference estimate of f'(x). Truncation error is O(h).

    Parameters
    ----------
    f : callable
        Function of one variable.
    x : float
        Point at which to estimate the derivative.
    h : float
        Step size (must be positive).

    Returns
    -------
    float
        Estimate of f'(x).
    """
    return (f(x + h) - f(x)) / h


def central_difference(f, x, h):
    """Central-difference estimate of f'(x). Truncation error is O(h**2).

    Parameters
    ----------
    f : callable
        Function of one variable.
    x : float
        Point at which to estimate the derivative.
    h : float
        Step size (must be positive).

    Returns
    -------
    float
        Estimate of f'(x).
    """
    return (f(x + h) - f(x - h)) / (2 * h)

# find area using point, point to point draw a line, then calculate area.
def trapezoid(f, a, b, n):
    """Integrate f from a to b with the composite trapezoid rule.

    Parameters
    ----------
    f : callable
        Function to integrate. Must accept a NumPy array.
    a, b : float
        Lower and upper limits.
    n : int
        Number of sub-intervals (must be at least 1).

    Returns
    -------
    float
        Estimate of the integral. Truncation error is O(h**2), h = (b - a) / n.

    Raises
    ------
    ValueError
        If n is less than 1.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    # End points count half; interior points count fully.
    return h * (y[0] / 2 + y[1:-1].sum() + y[-1] / 2)

# Calculate area using linear curve
def simpson(f, a, b, n):
    """Integrate f from a to b with the composite Simpson's 1/3 rule.

    Parameters
    ----------
    f : callable
        Function to integrate. Must accept a NumPy array.
    a, b : float
        Lower and upper limits.
    n : int
        Number of sub-intervals (must be even and at least 2).

    Returns
    -------
    float
        Estimate of the integral. Truncation error is O(h**4), h = (b - a) / n.

    Raises
    ------
    ValueError
        If n is odd or less than 2.
    """
    if n < 2 or n % 2 != 0:
        raise ValueError("n must be an even number of at least 2")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    # Weights 1, 4, 2, 4, ..., 2, 4, 1 from fitting parabolas through point triples.
    return h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])


def romberg(f, a, b, levels=5):
    """Integrate f from a to b with Romberg integration.

    Builds trapezoid estimates with 1, 2, 4, ... intervals, then repeatedly
    applies Richardson extrapolation to cancel the leading error terms.

    Parameters
    ----------
    f : callable
        Function to integrate. Must accept a NumPy array.
    a, b : float
        Lower and upper limits.
    levels : int, optional
        Number of refinement levels (default 5, i.e. up to 16 intervals).

    Returns
    -------
    float
        Most refined Romberg estimate of the integral.
    """
    R = np.zeros((levels, levels))
    for k in range(levels):
        R[k, 0] = trapezoid(f, a, b, 2**k)
        # Richardson extrapolation: combine two estimates to cancel error terms.
        for j in range(1, k + 1):
            R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)
    return R[levels - 1, levels - 1]

# Each point draw a rectangle from x axis, then calculate area
def gauss_legendre(f, a, b, n):
    """Integrate f from a to b with n-point Gauss-Legendre quadrature.

    Exact for polynomials up to degree 2n - 1.

    Parameters
    ----------
    f : callable
        Function to integrate. Must accept a NumPy array.
    a, b : float
        Lower and upper limits.
    n : int
        Number of sample points.

    Returns
    -------
    float
        Estimate of the integral.
    """
    # Standard nodes and weights on [-1, 1].
    nodes, weights = np.polynomial.legendre.leggauss(n)
    # Map the nodes from [-1, 1] to [a, b].
    x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    return 0.5 * (b - a) * np.sum(weights * f(x))


def solve_normal_equations(X, y):
    """Least-squares coefficients by solving the normal equations.

    Solves (X^T X) beta = X^T y. Simple, but forming X^T X squares the
    condition number, so it loses accuracy for ill-conditioned X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.
    y : ndarray, shape (n_samples,)
        Target values.

    Returns
    -------
    ndarray, shape (n_features,)
        Fitted coefficients.
    """
    return np.linalg.solve(X.T @ X, X.T @ y)


def solve_lstsq(X, y):
    """Least-squares coefficients using NumPy's SVD-based solver.

    Works on X directly, so it avoids squaring the condition number.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.
    y : ndarray, shape (n_samples,)
        Target values.

    Returns
    -------
    ndarray, shape (n_features,)
        Fitted coefficients.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def solve_ridge(X, y, alpha):
    """Ridge-regression coefficients: solves (X^T X + alpha*I) beta = X^T y.

    Adding alpha to the diagonal bounds the condition number and shrinks
    the coefficients. (For simplicity the intercept column is penalised too.)

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.
    y : ndarray, shape (n_samples,)
        Target values.
    alpha : float
        Regularisation strength (alpha >= 0; alpha = 0 gives the normal equations).

    Returns
    -------
    ndarray, shape (n_features,)
        Fitted coefficients.
    """
    n_features = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(n_features), X.T @ y)