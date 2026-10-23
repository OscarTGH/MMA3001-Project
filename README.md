# Airfoil Self-Noise Surrogate Model

Fast machine-learning prediction of the self-noise of a NACA 0012 blade
section from its flow and geometry conditions, trained and validated on NASA
wind-tunnel data. MMA3001 Numerical Methods and Machine Learning project.

## What it does

Airfoil self-noise matters for wind turbines, drone rotors and fans.
Measuring it needs an anechoic wind tunnel, and high-fidelity simulation can
take days per condition. This project builds a surrogate model that predicts
the noise level in milliseconds, so designers can run quick what-if studies
within the tested range.

**Inputs:** frequency (Hz), angle of attack (°), chord length (m), free-stream
velocity (m/s) and suction-side displacement thickness δ* (m).

**Output:** scaled sound pressure level (dB) at each frequency, plus a full
predicted spectrum and an overall noise level.

## Key results

Six models were compared using cross-validation that keeps whole wind-tunnel
set-ups together, so every score is on set-ups the model never saw.

| Model | RMSE, unseen set-ups (dB) | RMSE, unseen chord lengths (dB) |
|:--|:-:|:-:|
| Linear regression (baseline) | 4.84 | 4.88 |
| Physics scaling law (baseline) | 4.80 | 5.01 |
| Polynomial + ridge | 2.81 | 4.53 |
| Decision tree | 3.64 | 4.36 |
| SVR (RBF kernel) | 2.81 | 3.36 |
| **MLP neural network (final model)** | **2.61** | **3.19** |

The final MLP meets the 3 dB target (nested cross-validation), with 82% of
predictions within ±3 dB. Its errors concentrate at small chords with high
angles of attack, where it can be wrong by up to about 20 dB.

## Quick start

Requires Python 3.11 or later (developed with Python 3.14).

```bash
git clone https://github.com/OscarTGH/MMA3001-Project.git
cd MMA3001-Project
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pytest                          # all tests should pass
```

## Using the model

```python
from airfoil.predict import predict_spl, predict_spectrum, overall_level

# One prediction (dB)
predict_spl(frequency_hz=2000, alpha_deg=0, chord_m=0.3048,
            velocity_ms=55.5, delta_star_m=0.00283)

# Full 21-band spectrum and overall level for one condition
predict_spectrum(alpha_deg=0, chord_m=0.3048, velocity_ms=55.5, delta_star_m=0.00283)
overall_level(alpha_deg=0, chord_m=0.3048, velocity_ms=55.5, delta_star_m=0.00283)
```

Invalid inputs (missing, non-numeric or non-physical values) raise a
`ValueError`. Inputs outside the tested range raise an `ExtrapolationWarning`,
and small chords at high angle raise an `UnreliableRegionWarning`.

### Tested input ranges

| Input | Range |
|:--|:--|
| Frequency | 200 – 20,000 Hz |
| Angle of attack | 0 – 22.2° |
| Chord length | 0.0254 – 0.3048 m |
| Free-stream velocity | 31.7 – 71.3 m/s |
| Displacement thickness δ* | 0.0004 – 0.058 m |

## Reproducing the results

Run the notebooks in order. Each one saves results that later notebooks use.

| Notebook | What it does |
|:--|:--|
| `01_data_checks.ipynb` | Data validation: ranges, missing values, set-up count, spectra |
| `02_conditioning.ipynb` | Least-squares conditioning of polynomial regression |
| `03_baselines.ipynb` | Linear and physics-based baselines under three validation schemes |
| `04_ml_models.ipynb` | Polynomial, decision tree, SVR and MLP models, with overfitting experiments |
| `05_validation.ipynb` | Nested tuning, fold-by-fold model selection, validation figures and physical checks; saves `models/final_model.joblib` |
| `06_performance.ipynb` | Timing, profiling, FLOP counts, quadrature comparison and learning curve |

Figures and result tables are saved to `results/`. Timings depend on hardware,
so they may differ on another computer.

## Repository structure

```
├── data/          raw NASA dataset
├── docs/          HTML documentation of the source code (pdoc)
├── models/        saved final model
├── notebooks/     analysis notebooks (run in order)
├── results/       figures, result tables and test report
├── src/airfoil/   source code
│   ├── data.py         loading the dataset
│   ├── features.py     building model inputs
│   ├── numerics.py     dB sum, finite differences, quadrature, least squares
│   ├── physics.py      physics-based scaling-law model
│   ├── validation.py   set-up groups, metrics, cross-validation, nested CV
│   └── predict.py      prediction interface with input checks
├── tests/         pytest tests
├── requirements.txt
└── LICENSE
```

## Documentation and tests

- **Code documentation:** open `docs/index.html` in a browser (generated with
  pdoc from the docstrings).
- **Tests:** run `pytest`. The numerical routines are verified against
  analytical solutions, including order-of-accuracy checks for finite
  differences and quadrature. The latest test report is in `results/`.
- **Regenerate the documentation:** `pdoc ./src/airfoil --docformat numpy -o docs`

## Limitations

- Valid only for the NACA 0012 profile, at the span and observer position of
  the original experiments.
- Not validated outside the tested input ranges.
- Unreliable for small chords (≤ 0.05 m) at high angles of attack (> 15°).
- δ* must be supplied by the user, but in reality it depends on chord,
  velocity and angle of attack.
- The SPL values were scaled by NASA in an undocumented way, so predictions
  are only meaningful relative to this dataset.

## Data

NASA airfoil self-noise dataset, stored in `data/airfoil_self_noise.dat`.

Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5VW2C.
Licensed under CC BY 4.0.

Brooks, T. F., Pope, D. S., & Marcolini, M. A. (1989). *Airfoil Self-Noise and
Prediction*. NASA Reference Publication 1218.

## AI use

Claude (Anthropic) was used to explain concepts, review my plan, suggest code
and help interpret results. All code was run and checked against tests, and
all design decisions were mine. See the AI reflection in the project report.

## Licence

Code released under the MIT License (see `LICENSE`). The dataset keeps its
original CC BY 4.0 licence.

## Author

Tham Guo Hao 34478337