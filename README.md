# Numerical Analysis Methods in Python

A collection of five numerical analysis modules covering interpolation,
root/intersection finding, integration, curve fitting, and shape reconstruction.
Each module implements its algorithm from first principles (no black-box solver
calls) under explicit constraints on function-call budgets, precision, and
running time.

## Modules

| File | Description |
|---|---|
| `interpolation.py` | Interpolates a black-box function over `[a, b]` using at most `n` samples, via an O(n) natural cubic spline built with the Thomas algorithm. |
| `function_intersections.py` | Finds intersection points of two functions on `[a, b]` using a segment scan combined with a hybrid secant/bisection root solver. |
| `area_between_curves.py` | Computes definite integrals (float32 precision, Gauss-Legendre quadrature) and the enclosed area between two functions, built on `function_intersections.py`. |
| `curve_fitting.py` | Fits a polynomial model to noisy sampled data within a running-time budget, using Chebyshev/random sampling and a custom Gaussian-elimination least-squares solver. |
| `shape_fitting.py` | Reconstructs a closed shape's contour from noisy `(x, y)` samples using Fourier descriptors (via FFT), then computes its area. |

## Running the tests

The project uses `pytest`. Run the full suite from the project root:

```bash
pip install numpy scipy tqdm pytest
pytest -v
```

`test_core.py` is a compact suite covering the key behaviour of every module:
accuracy on known functions, function-call budgets, time limits, float32 return
types and edge cases (single sample point, no intersections, fewer than two
intersections, noisy samples).

Each module also has its own dedicated test file (`test_interpolation.py`,
`test_function_intersections.py`, `test_area_between_curves.py`,
`test_curve_fitting.py`, `test_shape_fitting.py`) that can be run individually,
e.g. `pytest test_interpolation.py -v`.

**Dependencies:** `numpy`, `tqdm` and `pytest` are required throughout. `test_area_between_curves.py`
additionally imports `scipy` (used only to compute reference values for the tests, not
by the solution code itself).

## Test infrastructure

`functionUtils.py`, `sampleFunctions.py`, `commons.py`, and `grader.py` are course-provided
infrastructure — decorators and helper functions for generating test functions/shapes,
restricting function-call counts, injecting noise/delay, and running an automated grading
pass. They are not part of the student's own solution code but are included here since the
five modules above depend on them to run and be tested.
