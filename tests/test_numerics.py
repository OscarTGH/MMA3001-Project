"""Verification tests for airfoil.numerics against known answers."""

import numpy as np
import pytest

from airfoil.numerics import (
    central_difference,
    db_sum,
    forward_difference,
    gauss_legendre,
    romberg,
    simpson,
    solve_lstsq,
    solve_normal_equations,
    solve_ridge,
    trapezoid,
)

# dB energy sum
# Test for combining two sounds of 100 dB each.
def test_db_sum_two_equal_levels():
    """Two equal levels add 10*log10(2) = 3.0103 dB."""
    assert db_sum([100, 100]) == pytest.approx(103.0103, abs=1e-4)

# Test for passing in just one level, 90 dB.
def test_db_sum_single_level_unchanged():
    """One level combined with nothing stays the same."""
    assert db_sum([90]) == pytest.approx(90)

# Test for combining a loud sound (100 dB) with a much quieter one (80 dB).
def test_db_sum_quieter_source_adds_little():
    """A source 20 dB quieter adds only 10*log10(1.01) = 0.0432 dB."""
    assert db_sum([100, 80]) == pytest.approx(100.0432, abs=1e-4)

# Test for extremely large levels, physically unrealistic and chosen on purpose.
def test_db_sum_no_overflow_for_huge_levels():
    """The max-factoring trick avoids overflow (naive method gives inf)."""
    result = db_sum([5000, 5000])
    assert np.isfinite(result)
    assert result == pytest.approx(5003.0103, abs=1e-4)

# Test for passing an empty list, no sound levels at all.
def test_db_sum_empty_raises():
    """An empty input is rejected."""
    with pytest.raises(ValueError):
        db_sum([])

# Finite differences
def test_central_difference_matches_cos():
    """d/dx sin(x) at x = 1 equals cos(1)."""
    estimate = central_difference(np.sin, 1.0, h=1e-5)
    assert estimate == pytest.approx(np.cos(1.0), abs=1e-8)


def test_forward_difference_is_first_order():
    """Shrinking h by 10 should shrink the error by about 10 (O(h))."""
    exact = np.cos(1.0)
    err_big = abs(forward_difference(np.sin, 1.0, 1e-2) - exact)
    err_small = abs(forward_difference(np.sin, 1.0, 1e-3) - exact)
    assert 5 < err_big / err_small < 20


def test_central_difference_is_second_order():
    """Shrinking h by 10 should shrink the error by about 100 (O(h**2))."""
    exact = np.cos(1.0)
    err_big = abs(central_difference(np.sin, 1.0, 1e-2) - exact)
    err_small = abs(central_difference(np.sin, 1.0, 1e-3) - exact)
    assert 50 < err_big / err_small < 200

# Trapizium and Simpson
def test_trapezoid_exact_for_straight_line():
    """Trapezoid is exact for a straight line: integral of 2x + 1 on [0, 1] = 2."""
    assert trapezoid(lambda x: 2 * x + 1, 0, 1, n=1) == pytest.approx(2.0)


def test_trapezoid_is_second_order():
    """Halving h should cut the error by about 4 (O(h**2))."""
    exact = 0.25  # integral of x**3 on [0, 1]
    err_big = abs(trapezoid(lambda x: x**3, 0, 1, 10) - exact)
    err_small = abs(trapezoid(lambda x: x**3, 0, 1, 20) - exact)
    assert 3.5 < err_big / err_small < 4.5


def test_simpson_exact_for_cubic():
    """Simpson is exact for cubics: integral of x**3 on [0, 1] = 0.25."""
    assert simpson(lambda x: x**3, 0, 1, n=2) == pytest.approx(0.25)


def test_simpson_is_fourth_order():
    """Halving h should cut the error by about 16 (O(h**4))."""
    exact = 2.0  # integral of sin(x) on [0, pi]
    err_big = abs(simpson(np.sin, 0, np.pi, 10) - exact)
    err_small = abs(simpson(np.sin, 0, np.pi, 20) - exact)
    assert 14 < err_big / err_small < 18


def test_simpson_odd_n_raises():
    """Simpson needs an even number of intervals."""
    with pytest.raises(ValueError):
        simpson(np.sin, 0, np.pi, n=3)


# Romberg and Gauss-Legendre
def test_romberg_accurate_for_sin():
    """Romberg with 5 levels (16 intervals) gives integral of sin on [0, pi] = 2."""
    assert romberg(np.sin, 0, np.pi, levels=5) == pytest.approx(2.0, abs=1e-7)


def test_gauss_two_points_exact_for_cubic():
    """2-point Gauss is exact up to degree 3: integral of x**3 on [0, 1] = 0.25."""
    assert gauss_legendre(lambda x: x**3, 0, 1, n=2) == pytest.approx(0.25)


def test_gauss_three_points_exact_for_degree_five():
    """3-point Gauss is exact up to degree 5: integral of x**5 on [0, 1] = 1/6."""
    assert gauss_legendre(lambda x: x**5, 0, 1, n=3) == pytest.approx(1 / 6)


def test_gauss_accurate_for_sin():
    """5-point Gauss gives integral of sin on [0, pi] = 2 to high accuracy."""
    assert gauss_legendre(np.sin, 0, np.pi, n=5) == pytest.approx(2.0, abs=1e-6)


# Least squares and conditioning
def test_solvers_recover_known_line():
    """Both solvers recover y = 2 + 3x exactly from noise-free data."""
    x = np.linspace(0, 1, 20)
    X = np.column_stack([np.ones_like(x), x])
    y = 2 + 3 * x
    assert solve_normal_equations(X, y) == pytest.approx([2, 3], abs=1e-10)
    assert solve_lstsq(X, y) == pytest.approx([2, 3], abs=1e-10)


def test_normal_equations_square_condition_number():
    """cond(X^T X) equals cond(X) squared."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    assert np.linalg.cond(X.T @ X) == pytest.approx(np.linalg.cond(X) ** 2, rel=1e-6)


def test_lstsq_more_accurate_when_ill_conditioned():
    """For an ill-conditioned polynomial fit, lstsq beats the normal equations."""
    x = np.linspace(0, 1, 50)
    X = np.vander(x, 10, increasing=True)  # columns 1, x, x**2, ..., x**9
    true_beta = np.ones(10)
    y = X @ true_beta
    err_normal = np.max(np.abs(solve_normal_equations(X, y) - true_beta))
    err_lstsq = np.max(np.abs(solve_lstsq(X, y) - true_beta))
    assert err_lstsq * 100 < err_normal


def test_ridge_improves_conditioning():
    """Adding alpha to the diagonal lowers the condition number."""
    x = np.linspace(0, 1, 50)
    X = np.vander(x, 10, increasing=True)
    cond_plain = np.linalg.cond(X.T @ X)
    cond_ridge = np.linalg.cond(X.T @ X + 1e-3 * np.eye(10))
    assert cond_ridge < cond_plain