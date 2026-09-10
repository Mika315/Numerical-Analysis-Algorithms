"""
Utility functions for extended numerical analysis testing
"""

import numpy as np
import time
from typing import Callable, List, Tuple, Dict
from functools import wraps
import threading
import _thread as thread
import traceback

# Set random seed for reproducibility
np.random.seed(42)

class TestTimer:
    """Context manager for timing code execution"""
    def __init__(self):
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


class TestReport:
    """Collects and formats test results"""
    def __init__(self):
        self.results = []
        self.timings = []
        self.task_scores = {i: {'passed': 0, 'total': 0} for i in range(1, 6)}

    def add_result(self, task: int, test_name: str, passed: bool,
                   error_msg: str = "", expected=None, actual=None,
                   elapsed_time: float = 0, hint: str = ""):
        """Add a test result"""
        self.results.append({
            'task': task,
            'test_name': test_name,
            'passed': passed,
            'error_msg': error_msg,
            'expected': expected,
            'actual': actual,
            'time': elapsed_time,
            'hint': hint
        })

        self.task_scores[task]['total'] += 1
        if passed:
            self.task_scores[task]['passed'] += 1

        if elapsed_time > 0:
            self.timings.append((test_name, elapsed_time))

    def print_summary(self):
        """Print comprehensive test summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])

        print("\n" + "="*80)
        print("TEST SUMMARY REPORT")
        print("="*80)

        # Overall statistics
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {100*passed_tests/total_tests:.2f}%")

        # Score calculation (out of 100)
        # Task weights: Task1=10, Task2=10, Task3=10, Task4=10, Task5=10
        task_weights = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10}
        total_score = 0

        print("\n" + "-"*80)
        print("SCORE BREAKDOWN BY TASK:")
        print("-"*80)

        for task in range(1, 6):
            stats = self.task_scores[task]
            if stats['total'] > 0:
                task_rate = stats['passed'] / stats['total']
                task_score = task_weights[task] * task_rate
                total_score += task_score
                print(f"Task {task}: {stats['passed']}/{stats['total']} passed "
                      f"({100*task_rate:.1f}%) - Score: {task_score:.2f}/{task_weights[task]}")
            else:
                print(f"Task {task}: No tests run")

        print("-"*80)
        print(f"TOTAL SCORE: {total_score:.2f}/50.0")
        print("="*80)

        # Slowest tests
        if self.timings:
            print("\nSLOWEST TESTS (Top 10):")
            print("-"*80)
            sorted_timings = sorted(self.timings, key=lambda x: x[1], reverse=True)[:10]
            for test_name, elapsed in sorted_timings:
                print(f"{test_name:60s} {elapsed:8.4f}s")

        # Most common failures
        failures = [r for r in self.results if not r['passed']]
        if failures:
            print("\nFAILED TESTS:")
            print("-"*80)
            for fail in failures[:20]:  # Show first 20 failures
                print(f"\nTest: {fail['test_name']}")
                print(f"Task: {fail['task']}")
                print(f"Error: {fail['error_msg']}")
                if fail['expected'] is not None:
                    print(f"Expected: {fail['expected']}")
                if fail['actual'] is not None:
                    print(f"Actual: {fail['actual']}")
                if fail['hint']:
                    print(f"Hint: {fail['hint']}")
                print(f"Time: {fail['time']:.4f}s")

        print("\n" + "="*80)


# ============================================================================
# FUNCTION GENERATORS
# ============================================================================

def generate_test_functions() -> List[Tuple[Callable, str, float, float]]:
    """
    Generate 50+ diverse test functions
    Returns: List of (function, name, a, b) tuples
    """
    functions = []

    # 1-10: Low degree polynomials
    for i in range(10):
        coeffs = np.random.uniform(-1, 1, np.random.randint(2, 5))
        poly = np.poly1d(coeffs)
        functions.append((poly, f"poly_low_deg_{i}", -10, 10))

    # 11-20: High degree polynomials
    for i in range(10):
        coeffs = np.random.uniform(-0.5, 0.5, np.random.randint(8, 15))
        poly = np.poly1d(coeffs)
        functions.append((poly, f"poly_high_deg_{i}", -5, 5))

    # 21-25: Trigonometric functions
    functions.extend([
        (lambda x: np.sin(x), "sin", 0, 2*np.pi),
        (lambda x: np.cos(x), "cos", 0, 2*np.pi),
        (lambda x: np.sin(x**2), "sin_x2", -3, 3),
        (lambda x: np.cos(2*x) + np.sin(x), "cos2x_sinx", 0, 2*np.pi),
        (lambda x: np.tan(x), "tan", -1.5, 1.5),
    ])

    # 26-30: Exponential functions
    functions.extend([
        (lambda x: np.exp(x), "exp", -2, 2),
        (lambda x: np.exp(-x**2), "gaussian", -3, 3),
        (lambda x: np.exp(np.sin(x)), "exp_sin", 0, 2*np.pi),
        (lambda x: 2**x, "2_pow_x", -5, 5),
        (lambda x: np.exp(-abs(x)), "exp_abs", -5, 5),
    ])

    # 31-35: Logarithmic functions
    functions.extend([
        (lambda x: np.log(x), "log", 0.1, 10),
        (lambda x: np.log(1 + x**2), "log_1_x2", -5, 5),
        (lambda x: x * np.log(x) if x > 0 else 0, "x_log_x", 0.1, 5),
        (lambda x: np.log(abs(np.sin(x)) + 1), "log_abs_sin", 0.1, 6),
        (lambda x: np.log10(x), "log10", 1, 100),
    ])

    # 36-40: Rational functions
    functions.extend([
        (lambda x: 1/(1 + x**2), "rational_1", -10, 10),
        (lambda x: x/(1 + x**2), "rational_2", -5, 5),
        (lambda x: (x**2 - 1)/(x**2 + 1), "rational_3", -10, 10),
        (lambda x: 1/np.sqrt(1 + x**2), "rational_4", -5, 5),
        (lambda x: (x**3 - x)/(x**2 + 2), "rational_5", -3, 3),
    ])

    # 41-45: Oscillating functions
    functions.extend([
        (lambda x: np.sin(10*x), "high_freq_sin", 0, 1),
        (lambda x: np.sin(x)/x if x != 0 else 1, "sinc", -10, 10),
        (lambda x: x * np.sin(1/x) if x != 0 else 0, "x_sin_1_x", -1, 1),
        (lambda x: np.sin(x) * np.exp(-x), "damped_sin", 0, 10),
        (lambda x: np.cos(x**2), "cos_x2", -3, 3),
    ])

    # 46-50: Special/challenging functions
    functions.extend([
        (lambda x: abs(x), "abs", -5, 5),
        (lambda x: np.sqrt(abs(x)), "sqrt_abs", -4, 4),
        (lambda x: x**3 - 3*x**2 + 2*x, "cubic_multiple_roots", -1, 3),
        (lambda x: np.arctan(x), "arctan", -10, 10),
        (lambda x: np.exp(x) * np.sin(x), "exp_sin_product", 0, 3),
    ])

    # ============================================================================
    # NEW ADDITIONS FOR TASK 4 (ONLY ADDING, NOT REMOVING)
    # ============================================================================

    def add_gaussian_noise(f: Callable, noise_level: float) -> Callable:
        """Wraps a function with Gaussian noise."""
        return lambda x: f(x) + np.random.normal(0, noise_level)

    def calculate_mse(f1: Callable, f2: Callable, a: float, b: float, n_points: int = 500) -> float:
        """Calculates Mean Squared Error between two functions."""
        xs = np.linspace(a, b, n_points)
        mse = np.mean([(f1(x) - f2(x)) ** 2 for x in xs])
        return mse

    def generate_task4_cases():
        """Returns a list of clean functions and their parameters for Task 4."""
        return [
            (lambda x: 2 * x + 3, 0, 10, 1, 0.5, "Linear Basic"),
            (lambda x: x ** 2 - 5 * x + 6, -5, 5, 2, 1.0, "Quadratic Noise"),
            (np.poly1d([1, 0, -2, 4]), -3, 3, 3, 0.8, "Cubic Polynomial"),
            (lambda x: np.sin(x), 0, np.pi, 4, 0.2, "Sine Approx"),
            (lambda x: np.exp(x / 5), 0, 5, 3, 0.5, "Exp Approx")
        ]

    # 51-55: Piecewise and step functions
    def piecewise1(x):
        if x < 0:
            return x**2
        else:
            return np.sqrt(x)

    def piecewise2(x):
        if x < -1:
            return -1
        elif x < 1:
            return x
        else:
            return 1

    functions.extend([
        (np.vectorize(piecewise1), "piecewise_1", -2, 2),
        (np.vectorize(piecewise2), "piecewise_2", -2, 2),
        (lambda x: np.sign(x), "sign", -5, 5),
        (lambda x: np.floor(x), "floor", -3, 3),
        (lambda x: x - np.floor(x), "fractional", -5, 5),
    ])

    return functions


def generate_intersection_pairs() -> List[Tuple[Callable, Callable, str, float, float, List[float]]]:
    """
    Generate pairs of functions with known intersections
    Returns: List of (f1, f2, name, a, b, expected_roots) tuples
    """
    pairs = []

    # Simple polynomial intersections
    pairs.append((
        lambda x: x**2,
        lambda x: x + 2,
        "parabola_line",
        -3, 3,
        [-1.0, 2.0]
    ))

    pairs.append((
        lambda x: x**2 - 4,
        lambda x: 0*x,
        "parabola_x_axis",
        -3, 3,
        [-2.0, 2.0]
    ))

    # Trigonometric intersections
    pairs.append((
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        "sin_cos",
        -np.pi, np.pi,
        [-3*np.pi/4, np.pi/4]
    ))

    # Multiple intersections
    pairs.append((
        lambda x: np.sin(x),
        lambda x: 0*x,
        "sin_zero_multiple",
        -2*np.pi, 2*np.pi,
        [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi]
    ))

    # Close intersections
    pairs.append((
        lambda x: x**3,
        lambda x: 3*x**2 - 2*x,
        "close_intersections",
        -1, 3,
        [0.0, 1.0, 2.0]
    ))

    # Tangent intersection
    pairs.append((
        lambda x: x**2,
        lambda x: 2*x - 1,
        "tangent",
        0, 2,
        [1.0]
    ))

    # Exponential vs polynomial
    pairs.append((
        lambda x: np.exp(x),
        lambda x: x + 1,
        "exp_linear",
        -2, 2,
        [0.0]
    ))

    # High frequency oscillations
    pairs.append((
        lambda x: np.sin(10*x),
        lambda x: 0.5*x,
        "high_freq_line",
        0, 2,
        []  # Many intersections, hard to list all
    ))

    return pairs
# --- Additions to utils.py (in addition to existing) ---

def calculate_relative_error(f_fitted: Callable, f_true: Callable, a: float, b: float, n_points: int = 500) -> float:
    """
    Calculates the average relative error: |f_fitted - f_true| / (|f_true| + epsilon)
    """
    xs = np.linspace(a, b, n_points)
    epsilon = 1e-10  # to avoid division by zero
    errors = []
    for x in xs:
        y_true = f_true(x)
        y_fit = f_fitted(x)
        errors.append(abs(y_fit - y_true) / (abs(y_true) + epsilon))
    return float(np.mean(errors))

def add_outlier_noise(f: Callable, noise_level: float = 0.1, outlier_prob: float = 0.05):
    """
    Adds Gaussian noise and occasional extreme outliers to the function.
    """
    def noisy_f(x):
        val = f(x) + np.random.normal(0, noise_level)
        if np.random.random() < outlier_prob:
            val += np.random.choice([-10, 10]) * noise_level * 50  # extreme outlier
        return val
    return noisy_f

def generate_hard_task4_cases():
    """
    Returns complex, non-polynomial or high-frequency functions for Task 4.
    """
    return [
        (lambda x: np.sin(10 * x), 0, 2, 8, 0.2, "High Frequency Sine"),
        (lambda x: 1 / (1 + x**2), -2, 2, 6, 0.1, "Runge's Function (Rational)"),
        (lambda x: np.abs(x), -1, 1, 4, 0.05, "Abs Function (Non-differentiable)"),
        (lambda x: np.log(x + 1.1), 0, 5, 3, 0.1, "Logarithmic Growth"),
        (lambda x: np.exp(-x**2), -2, 2, 6, 0.05, "Gaussian Bell Curve")
    ]


def check_interpolation_error(f_interpolated: Callable, f_true: Callable,
                               a: float, b: float, n_test: int = 100) -> Tuple[float, float]:
    """
    Check interpolation error
    Returns: (mean_abs_error, max_abs_error)
    """
    test_points = np.linspace(a, b, n_test)
    errors = []

    for x in test_points:
        try:
            y_true = f_true(x)
            y_interp = f_interpolated(x)
            errors.append(abs(y_true - y_interp))
        except:
            errors.append(float('inf'))

    return np.mean(errors), np.max(errors)


def check_integration_error(result: float, expected: float,
                            rel_tol: float = 0.01, abs_tol: float = 0.01) -> Tuple[bool, float, str]:
    """
    Check integration accuracy
    Returns: (is_accurate, error, hint)
    """
    abs_error = abs(result - expected)

    if expected != 0:
        rel_error = abs_error / abs(expected)
    else:
        rel_error = abs_error

    is_accurate = (rel_error < rel_tol) or (abs_error < abs_tol)

    hint = ""
    if not is_accurate:
        if abs_error > 10:
            hint = "Large absolute error - check integration bounds or method"
        elif rel_error > 0.5:
            hint = "High relative error - likely wrong integration method or too few points"
        else:
            hint = "Moderate error - consider using more points or adaptive method"

    return is_accurate, abs_error, hint
def quit_function(fn_name):
    print(f'{fn_name} took too long', file=sys.stderr)
    sys.stderr.flush()
    thread.interrupt_main()

def exit_after(s):
    """Decorator to stop a function if it exceeds the time limit, as used in grader.py."""
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            try:
                timer.start()
                return fn(*args, **kwargs)
            finally:
                timer.cancel()
        return inner
    return outer

def calculate_relative_error(fres, fexp, a, b, n=200):
    """Computes the average relative error at n points."""
    xs = np.linspace(a, b, n)
    eps = 1e-10  # to avoid division by zero
    errors = []
    for x in xs:
        actual = fexp(x)
        fitted = fres(x)
        errors.append(abs(fitted - actual) / (abs(actual) + eps))
    return float(np.mean(errors))

def hard_functions_generator():
    """Especially challenging functions for fitting."""
    return [
        # Runge's function - very hard for polynomials due to edge oscillations
        (lambda x: 1 / (1 + 25 * x**2), -1, 1, 10, "Runge_Function"),
        # High-frequency oscillations
        (lambda x: np.sin(x**2), 0, 3, 12, "High_Freq_Oscillation"),
        # Function with a discontinuous derivative (sharp turn)
        (lambda x: np.abs(x) + x**2, -2, 2, 4, "Abs_Plus_Quadratic")
    ]


def count_function_calls(func: Callable) -> Tuple[Callable, object]:
    """
    Wrap a function to count calls
    Returns: (wrapped_function, counter_object)
    """
    class Counter:
        def __init__(self):
            self.count = 0

    counter = Counter()

    @wraps(func)
    def wrapper(*args, **kwargs):
        counter.count += 1
        return func(*args, **kwargs)

    return wrapper, counter
