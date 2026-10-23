"""Surrogate models for predicting airfoil self-noise (MMA3001 project).

Modules
-------
- ``data``: load the NASA airfoil self-noise dataset.
- ``features``: build model input matrices.
- ``numerics``: dB energy sum, finite differences, quadrature and
  least-squares solvers, verified against analytical solutions.
- ``physics``: physics-based scaling-law baseline model.
- ``validation``: set-up groups, error metrics, cross-validation and
  nested cross-validation.
- ``predict``: prediction interface for the final model, with input
  checks and warnings.

Quick example
-------------
>>> from airfoil.predict import predict_spl
>>> predict_spl(2000, 0, 0.3048, 55.5, 0.00283)   # SPL in dB
"""