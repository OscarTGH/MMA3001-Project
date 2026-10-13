"""Verification tests for airfoil.numerics against known answers."""

import numpy as np
import pytest

from airfoil.numerics import db_sum

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