"""
Extended tests for Assignment 1: Interpolation
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (TestTimer, TestReport, generate_test_functions,
                   check_interpolation_error, count_function_calls)

# Import the assignment
try:
    from interpolation import Assignment1
except ImportError:
    print("ERROR: Could not import interpolation.py")
    print("Make sure interpolation.py is in the parent directory")
    sys.exit(1)


class TestAssignment1:
    """Comprehensive tests for interpolation"""

    @classmethod
    def setup_class(cls):
        """Initialize test suite"""
        cls.assignment = Assignment1()
        cls.report = TestReport()

    @classmethod
    def teardown_class(cls):
        """Print summary after all tests"""
        cls.report.print_summary()

    def test_basic_linear_function(self):
        """Test 1.1: Linear function y=2x+1"""
        f = lambda x: 2*x + 1

        with TestTimer() as timer:
            wrapped_f, counter = count_function_calls(f)
            f_interp = self.assignment.interpolate(wrapped_f, 0, 10, 20)

            # Check function call limit
            if counter.count > 20:
                self.report.add_result(
                    1, "test_basic_linear", False,
                    f"Called function {counter.count} times, limit is 20",
                    expected="≤20 calls", actual=f"{counter.count} calls",
                    elapsed_time=timer.elapsed,
                    hint="Reduce number of function evaluations"
                )
                pytest.fail(f"Too many function calls: {counter.count}")

            mean_err, max_err = check_interpolation_error(f_interp, f, 0, 10, 100)

        passed = mean_err < 0.01
        self.report.add_result(
            1, "test_basic_linear", passed,
            "" if passed else f"Mean error {mean_err:.6f} too high",
            expected="<0.01", actual=f"{mean_err:.6f}",
            elapsed_time=timer.elapsed,
            hint="Linear functions should interpolate with very low error"
        )
        assert passed, f"Mean error {mean_err:.6f} exceeds tolerance"

    def test_quadratic_function(self):
        """Test 1.2: Quadratic function y=x²-3x+5"""
        f = np.poly1d([1, -3, 5])

        with TestTimer() as timer:
            wrapped_f, counter = count_function_calls(f)
            f_interp = self.assignment.interpolate(wrapped_f, -5, 5, 30)
            mean_err, max_err = check_interpolation_error(f_interp, f, -5, 5, 100)

        passed = mean_err < 0.1
        self.report.add_result(
            1, "test_quadratic", passed,
            "" if passed else f"Mean error {mean_err:.6f} too high",
            expected="<0.1", actual=f"{mean_err:.6f}",
            elapsed_time=timer.elapsed,
            hint="Quadratic should be well approximated with 30 points"
        )
        assert passed

    def test_high_degree_polynomial(self):
        """Test 1.3: High degree polynomial (degree 10)"""
        coeffs = np.random.RandomState(42).uniform(-0.5, 0.5, 11)
        f = np.poly1d(coeffs)

        with TestTimer() as timer:
            wrapped_f, counter = count_function_calls(f)
            f_interp = self.assignment.interpolate(wrapped_f, -2, 2, 100)
            mean_err, max_err = check_interpolation_error(f_interp, f, -2, 2, 200)

        passed = mean_err < 1.0 and counter.count <= 100
        hint = ""
        if counter.count > 100:
            hint = "Exceeded function call limit"
        elif mean_err >= 1.0:
            hint = "High degree polynomials need more points or better method"

        self.report.add_result(
            1, "test_high_degree_poly", passed,
            "" if passed else f"Mean error {mean_err:.6f}, calls {counter.count}",
            expected="<1.0 error, ≤100 calls",
            actual=f"{mean_err:.6f} error, {counter.count} calls",
            elapsed_time=timer.elapsed,
            hint=hint
        )
        assert passed

    def test_sine_function(self):
        """Test 1.4: Sine function"""
        f = lambda x: np.sin(x)

        with TestTimer() as timer:
            f_interp = self.assignment.interpolate(f, 0, 2*np.pi, 50)
            mean_err, max_err = check_interpolation_error(f_interp, f, 0, 2*np.pi, 100)

        passed = mean_err < 0.1
        self.report.add_result(
            1, "test_sine", passed,
            "" if passed else f"Mean error {mean_err:.6f} too high for sine",
            expected="<0.1", actual=f"{mean_err:.6f}",
            elapsed_time=timer.elapsed,
            hint="Smooth periodic functions should interpolate well"
        )
        assert passed

    def test_exponential_function(self):
        """Test 1.5: Exponential function"""
        f = lambda x: np.exp(x)

        with TestTimer() as timer:
            f_interp = self.assignment.interpolate(f, -1, 1, 40)
            mean_err, max_err = check_interpolation_error(f_interp, f, -1, 1, 100)

        passed = mean_err < 0.1
        self.report.add_result(
            1, "test_exponential", passed,
            "" if passed else f"Mean error {mean_err:.6f}",
            expected="<0.1", actual=f"{mean_err:.6f}",
            elapsed_time=timer.elapsed,
            hint="Exponential grows quickly - may need non-uniform spacing"
        )
        assert passed

    def test_gaussian_function(self):
        """Test 1.6: Gaussian function e^(-x²)"""
        f = lambda x: np.exp(-x**2)

        with TestTimer() as timer:
            f_interp = self.assignment.interpolate(f, -3, 3, 50)
            mean_err, max_err = check_interpolation_error(f_interp, f, -3, 3, 100)

        passed = mean_err < 0.05
        self.report.add_result(
            1, "test_gaussian", passed,
            "" if passed else f"Mean error {mean_err:.6f}",
            expected="<0.05", actual=f"{mean_err:.6f}",
            elapsed_time=timer.elapsed,
            hint="Smooth bell curve should interpolate with low error"
        )
        assert passed

    def test_rational_function(self):
        """Test 1.7: Rational function 1/(1+x²)"""
        f = lambda x: 1/(1 + x**2)

        with TestTimer() as timer:
            f_interp = self.assignment.interpolate(f, -5, 5, 60)
            mean_err, max_err = check_interpolation_error(f_interp, f, -5, 5, 100)

        passed = mean_err < 0.1
        self.report.add_result(
            1, "test_rational", passed,
            "" if passed else f"Mean error {mean_err:.6f}",
            expected="<0.1", actual=f"{mean_err:.6f}",
            elapsed_time=timer.elapsed,
            hint="Rational functions may need careful point placement"
        )
        assert passed

    def test_small_interval(self):
        """Test 1.8: Very small interval [0, 0.001]"""
        f = lambda x: x**2 + x

        with TestTimer() as timer:
            f_interp = self.assignment.interpolate(f, 0, 0.001, 10)
            mean_err, max_err = check_interpolation_error(f_interp, f, 0, 0.001, 50)

        passed = mean_err < 1e-6
        self.report.add_result(
            1, "test_small_interval", passed,
            "" if passed else f"Mean error {mean_err:.9f} on tiny interval",
            expected="<1e-6", actual=f"{mean_err:.9f}",
            elapsed_time=timer.elapsed,
            hint="Small intervals need careful floating point handling"
        )
        assert passed

    def test_large_range(self):
        """Test 1.9: Large range [-100, 100]"""
        f = lambda x: 0.01*x**2 + x

        with TestTimer() as timer:
            f_interp = self.assignment.interpolate(f, -100, 100, 100)
            mean_err, max_err = check_interpolation_error(f_interp, f, -100, 100, 200)

        passed = mean_err < 10
        self.report.add_result(
            1, "test_large_range", passed,
            "" if passed else f"Mean error {mean_err:.6f}",
            expected="<10", actual=f"{mean_err:.6f}",
            elapsed_time=timer.elapsed,
            hint="Large ranges may accumulate numerical errors"
        )
        assert passed

    def test_oscillating_function(self):
        """Test 1.10: High frequency oscillation sin(10x)"""
        f = lambda x: np.sin(10*x)

        with TestTimer() as timer:
            f_interp = self.assignment.interpolate(f, 0, 2, 100)
            mean_err, max_err = check_interpolation_error(f_interp, f, 0, 2, 200)

        passed = mean_err < 0.5
        self.report.add_result(
            1, "test_oscillating", passed,
            "" if passed else f"Mean error {mean_err:.6f} for high frequency",
            expected="<0.5", actual=f"{mean_err:.6f}",
            elapsed_time=timer.elapsed,
            hint="High frequency functions need many points - Nyquist criterion"
        )
        assert passed

    def test_discontinuous_derivative(self):
        """Test 1.11: Absolute value function |x|"""
        f = lambda x: np.abs(x)

        with TestTimer() as timer:
            f_interp = self.assignment.interpolate(f, -5, 5, 50)
            mean_err, max_err = check_interpolation_error(f_interp, f, -5, 5, 100)

        passed = mean_err < 0.5
        self.report.add_result(
            1, "test_abs_function", passed,
            "" if passed else f"Mean error {mean_err:.6f}",
            expected="<0.5", actual=f"{mean_err:.6f}",
            elapsed_time=timer.elapsed,
            hint="Non-smooth functions have higher interpolation error"
        )
        assert passed

    def test_minimum_points(self):
        """Test 1.12: Minimum points (n=1)"""
        f = lambda x: 5.0

        with TestTimer() as timer:
            wrapped_f, counter = count_function_calls(f)
            f_interp = self.assignment.interpolate(wrapped_f, 0, 10, 1)

            # For constant function, should return constant
            try:
                result = f_interp(5.0)
                passed = abs(result - 5.0) < 0.1
                error_msg = "" if passed else f"Expected 5.0, got {result}"
            except:
                passed = False
                error_msg = "Exception thrown"

        self.report.add_result(
            1, "test_minimum_points", passed,
            error_msg,
            expected="5.0", actual=f"{result if 'result' in locals() else 'N/A'}",
            elapsed_time=timer.elapsed,
            hint="Handle edge case of n=1"
        )
        assert passed

    def test_runtime_complexity_on(self):
        """Test 1.13: Runtime complexity O(n) check"""
        f = lambda x: x**2
        times = []

        for n in [50, 100, 200]:
            with TestTimer() as timer:
                f_interp = self.assignment.interpolate(f, 0, 10, n)
                _ = f_interp(5.0)  # Make sure it works
            times.append(timer.elapsed)

        # For O(n), doubling n should roughly double time
        # For O(n²), doubling n should quadruple time
        ratio1 = times[1] / max(times[0], 1e-6)
        ratio2 = times[2] / max(times[1], 1e-6)

        # If O(n²), these ratios would be ~4. If O(n), ~2
        is_linear = ratio1 < 3 and ratio2 < 3

        self.report.add_result(
            1, "test_runtime_on", is_linear,
            "" if is_linear else f"Ratios {ratio1:.2f}, {ratio2:.2f} suggest O(n²)",
            expected="O(n) scaling",
            actual=f"Ratios: {ratio1:.2f}, {ratio2:.2f}",
            elapsed_time=sum(times),
            hint="O(n²) detected - optimize to O(n) for full points"
        )
        # Don't fail, just report

    def test_random_polynomials(self):
        """Test 1.14-1.23: 10 random polynomials"""
        np.random.seed(100)

        for i in range(10):
            degree = np.random.randint(2, 8)
            coeffs = np.random.uniform(-1, 1, degree + 1)
            f = np.poly1d(coeffs)

            with TestTimer() as timer:
                f_interp = self.assignment.interpolate(f, -5, 5, 50)
                mean_err, max_err = check_interpolation_error(f_interp, f, -5, 5, 100)

            passed = mean_err < 0.5
            self.report.add_result(
                1, f"test_random_poly_{i}", passed,
                "" if passed else f"Mean error {mean_err:.6f}",
                expected="<0.5", actual=f"{mean_err:.6f}",
                elapsed_time=timer.elapsed,
                hint=f"Random polynomial degree {degree}"
            )
            assert passed, f"Random polynomial {i} failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
