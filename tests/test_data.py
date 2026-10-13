"""Tests for airfoil.data."""

import pytest

from airfoil.data import COLUMNS, load_data


def test_shape():
    """The raw dataset has 1503 rows and 6 columns."""
    df = load_data()
    assert df.shape == (1503, 6)


def test_column_names():
    """Columns are named in the documented order."""
    df = load_data()
    assert list(df.columns) == COLUMNS


def test_no_missing_values():
    """The dataset contains no missing values."""
    df = load_data()
    assert df.isna().sum().sum() == 0


def test_missing_file_raises():
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data("does_not_exist.dat")