"""Data loading utilities for the NASA airfoil self-noise dataset."""

from pathlib import Path

import pandas as pd

# Column names in the order they appear in the raw file (units in the name).
COLUMNS = [
    "frequency_hz",   # third-octave band centre frequency [Hz]
    "alpha_deg",      # angle of attack [degrees]
    "chord_m",        # chord length [m]
    "velocity_ms",    # free-stream velocity [m/s]
    "delta_star_m",   # suction-side displacement thickness [m]
    "spl_db",         # scaled sound pressure level [dB] (target)
]

# Default location of the raw file: <project root>/data/airfoil_self_noise.dat
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "airfoil_self_noise.dat"


def load_data(path=DEFAULT_PATH):
    """Load the airfoil self-noise dataset into a DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Location of the tab-separated raw data file.
        Defaults to ``data/airfoil_self_noise.dat`` in the project root.

    Returns
    -------
    pandas.DataFrame
        One row per measured band level, with the columns listed in
        ``COLUMNS`` (SI units, angle in degrees, SPL in dB).

    Raises
    ------
    FileNotFoundError
        If no file exists at ``path``.
    """
    path = Path(path)

    # Fail early with a clear message rather than a confusing pandas error.
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # The raw file is tab-separated with no header row, so we supply names.
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS)

    return df