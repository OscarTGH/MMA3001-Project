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