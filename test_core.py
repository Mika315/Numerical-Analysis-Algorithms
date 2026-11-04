"""
Core pytest suite for the five numerical modules.
Run: pytest test_core.py -v
"""
import time

import numpy as np
import pytest

from interpolation import Assignment1
from function_intersections import Assignment2
from area_between_curves import Assignment3
from curve_fitting import Assignment4
from shape_fitting import Assignment5


# ---------- fixtures ----------

@pytest.fixture
def call_counter():
    """Wraps a function and counts how many times it is called (to check call budgets)."""
    def wrap(f):
        def counted(x):
            counted.calls += 1
            return f(x)
        counted.calls = 0
        return counted
    return wrap


# ---------- 1. interpolation ----------

@pytest.mark.parametrize("f, a, b", [
    (lambda x: 2 * x + 1, -5, 5),     # linear
    (lambda x: x ** 3 - x, -2, 2),    # polynomial
    (np.sin, 0, 2 * np.pi),           # smooth, non-polynomial
])
def test_interpolation_is_accurate(f, a, b):
    g = Assignment1().interpolate(f, a, b, n=50)
    xs = np.random.uniform(a, b, 200)
    mean_err = np.mean([abs(g(x) - f(x)) for x in xs])
    assert mean_err < 1e-3


def test_interpolation_respects_call_budget(call_counter):
    f = call_counter(np.cos)
    Assignment1().interpolate(f, 0, 10, n=20)
    assert f.calls <= 20


def test_interpolation_edge_case_single_point():
    g = Assignment1().interpolate(lambda x: 7.0, 0, 1, n=1)
    assert g(0.5) == pytest.approx(7.0)


# ---------- 2. intersections ----------

def test_intersections_are_real_roots():
    f1, f2 = (lambda x: x ** 2), (lambda x: 4.0)
    xs = Assignment2().intersections(f1, f2, -5, 5, maxerr=0.001)
    assert len(xs) == 2
    assert all(abs(f1(x) - f2(x)) <= 0.001 for x in xs)


def test_intersections_sin_cos():
    xs = Assignment2().intersections(np.sin, np.cos, -np.pi, np.pi, maxerr=0.001)
    assert any(x == pytest.approx(np.pi / 4, abs=0.01) for x in xs)
    assert any(x == pytest.approx(-3 * np.pi / 4, abs=0.01) for x in xs)


def test_intersections_none_when_curves_never_meet():
    xs = Assignment2().intersections(lambda x: x ** 2 + 1, lambda x: 0.0, -5, 5)
    assert len(xs) == 0


# ---------- 3. integration & area ----------

@pytest.mark.parametrize("f, a, b, expected", [
    (lambda x: x ** 2, 0, 3, 9.0),
    (np.sin, 0, np.pi, 2.0),
    (lambda x: 5.0, 2, 4, 10.0),
])
def test_integrate_known_values(f, a, b, expected):
    result = Assignment3().integrate(f, a, b, n=20)
    assert float(result) == pytest.approx(expected, rel=1e-3)


def test_integrate_returns_float32():
    assert isinstance(Assignment3().integrate(np.cos, 0, 1, n=10), np.float32)


def test_integrate_respects_call_budget(call_counter):
    f = call_counter(np.exp)
    Assignment3().integrate(f, 0, 1, n=10)
    assert f.calls <= 10


def test_area_between_curves():
    # x^2 and 6x-8 meet at x=2 and x=4; enclosed area = 4/3
    area = Assignment3().areabetween(lambda x: x ** 2, lambda x: 6 * x - 8)
    assert float(area) == pytest.approx(4 / 3, rel=1e-3)


def test_area_is_nan_with_less_than_two_intersections():
    area = Assignment3().areabetween(lambda x: x, lambda x: x + 1)
    assert np.isnan(area)


# ---------- 4. curve fitting (noisy data) ----------

def test_fit_noisy_line_within_time_limit():
    rng = np.random.default_rng(0)
    noisy = lambda x: 3 * x + 2 + rng.normal(0, 0.5)
    maxtime = 1.0
    start = time.time()
    g = Assignment4().fit(noisy, 0, 10, d=1, maxtime=maxtime)
    assert time.time() - start < maxtime
    assert g(5) == pytest.approx(17, abs=0.2)


# ---------- 5. shapes ----------

def circle(r):
    def contour(n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    return contour


def test_area_of_circle():
    assert float(Assignment5().area(circle(2))) == pytest.approx(np.pi * 4, rel=1e-3)


def test_fit_shape_from_noisy_samples():
    rng = np.random.default_rng(1)

    def sample():
        t = rng.uniform(0, 2 * np.pi)
        return 3 * np.cos(t) + rng.normal(0, 0.05), 3 * np.sin(t) + rng.normal(0, 0.05)

    shape = Assignment5().fit_shape(sample, maxtime=1.0)
    assert float(shape.area()) == pytest.approx(np.pi * 9, rel=0.05)
