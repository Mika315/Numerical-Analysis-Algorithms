"""
Extended tests for Assignment 3: Integration and Area Between Curves
"""

import pytest
import numpy as np
import sys
import os
from scipy import integrate as scipy_integrate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import TestTimer, TestReport, count_function_calls

try:
    from area_between_curves import Assignment3
except ImportError:
    print("ERROR: Could not import area_between_curves.py")
    sys.exit(1)


class TestAssignment3:
    """Comprehensive tests for integration"""

    @classmethod
    def setup_class(cls):
        cls.assignment = Assignment3()
        cls.report = TestReport()

    @classmethod
    def teardown_class(cls):
        cls.report.print_summary()

    def test_integrate_constant(self):
        """Test 3.1: Integrate constant function"""
        f = lambda x: 5.0
        expected = 5.0 * (10 - 2)  # 5 * 8 = 40

        with TestTimer() as timer:
            wrapped_f, counter = count_function_calls(f)
            result = self.assignment.integrate(wrapped_f, 2, 10, 20)

        # Check dtype
        is_float32 = result.dtype == np.float32
        error = abs(float(result) - expected)
        rel_error = error / expected if expected != 0 else error

        passed = is_float32 and rel_error < 0.01 and counter.count <= 20

        self.report.add_result(
            3, "test_integrate_constant", passed,
            "" if passed else f"Error: {error:.6f}, calls: {counter.count}",
            expected=f"{expected} (float32)",
            actual=f"{result} ({result.dtype})",
            elapsed_time=timer.elapsed,
            hint="Constant function integral should be exact"
        )
        assert passed

    def test_integrate_linear(self):
        """Test 3.2: Integrate linear function x"""
        f = lambda x: x
        # Integral of x from 0 to 10 = [x²/2] = 50
        expected = 50.0

        with TestTimer() as timer:
            result = self.assignment.integrate(f, 0, 10, 30)

        error = abs(float(result) - expected)
        rel_error = error / expected

        passed = rel_error < 0.01

        self.report.add_result(
            3, "test_integrate_linear", passed,
            "" if passed else f"Relative error: {rel_error:.6f}",
            expected=f"{expected}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Linear function should integrate with high accuracy"
        )
        assert passed

    def test_integrate_quadratic(self):
        """Test 3.3: Integrate x² from 0 to 3"""
        f = lambda x: x**2
        # [x³/3] from 0 to 3 = 9
        expected = 9.0

        with TestTimer() as timer:
            result = self.assignment.integrate(f, 0, 3, 50)

        error = abs(float(result) - expected)

        passed = error < 0.1

        self.report.add_result(
            3, "test_integrate_quadratic", passed,
            "" if passed else f"Error: {error:.6f}",
            expected=f"{expected}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Quadratic should have small error"
        )
        assert passed

    def test_integrate_polynomial(self):
        """Test 3.4: Integrate polynomial x³ - 2x² + x - 1"""
        f = np.poly1d([1, -2, 1, -1])
        # Antiderivative: x⁴/4 - 2x³/3 + x²/2 - x
        # From 0 to 2: 4 - 16/3 + 2 - 2 = 4 - 5.333 = -1.333
        expected = 2**4/4 - 2*2**3/3 + 2**2/2 - 2

        with TestTimer() as timer:
            wrapped_f, counter = count_function_calls(f)
            result = self.assignment.integrate(wrapped_f, 0, 2, 40)

        error = abs(float(result) - expected)

        passed = error < 0.1 and counter.count <= 40

        self.report.add_result(
            3, "test_integrate_polynomial", passed,
            "" if passed else f"Error: {error:.6f}, calls: {counter.count}",
            expected=f"{expected:.4f}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Higher degree polynomials may need more points"
        )
        assert passed

    def test_integrate_sine(self):
        """Test 3.5: Integrate sin(x) from 0 to π"""
        f = lambda x: np.sin(x)
        # [-cos(x)] from 0 to π = -(-1) - (-1) = 2
        expected = 2.0

        with TestTimer() as timer:
            result = self.assignment.integrate(f, 0, np.pi, 50)

        error = abs(float(result) - expected)

        passed = error < 0.01

        self.report.add_result(
            3, "test_integrate_sine", passed,
            "" if passed else f"Error: {error:.6f}",
            expected=f"{expected}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Periodic smooth functions should integrate accurately"
        )
        assert passed

    def test_integrate_exponential(self):
        """Test 3.6: Integrate e^x from 0 to 1"""
        f = lambda x: np.exp(x)
        # [e^x] from 0 to 1 = e - 1 ≈ 1.718
        expected = np.e - 1

        with TestTimer() as timer:
            result = self.assignment.integrate(f, 0, 1, 40)

        error = abs(float(result) - expected)
        rel_error = error / expected

        passed = rel_error < 0.01

        self.report.add_result(
            3, "test_integrate_exponential", passed,
            "" if passed else f"Rel error: {rel_error:.6f}",
            expected=f"{expected:.6f}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Exponential function integration"
        )
        assert passed

    def test_integrate_gaussian(self):
        """Test 3.7: Integrate e^(-x²) from -1 to 1"""
        f = lambda x: np.exp(-x**2)
        # No closed form, use scipy for reference
        expected, _ = scipy_integrate.quad(f, -1, 1)

        with TestTimer() as timer:
            result = self.assignment.integrate(f, -1, 1, 50)

        error = abs(float(result) - expected)
        rel_error = error / expected

        passed = rel_error < 0.01

        self.report.add_result(
            3, "test_integrate_gaussian", passed,
            "" if passed else f"Rel error: {rel_error:.6f}",
            expected=f"{expected:.6f}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Gaussian integral"
        )
        assert passed

    def test_integrate_rational(self):
        """Test 3.8: Integrate 1/(1+x²) from 0 to 1"""
        f = lambda x: 1/(1 + x**2)
        # [arctan(x)] from 0 to 1 = π/4
        expected = np.pi / 4

        with TestTimer() as timer:
            result = self.assignment.integrate(f, 0, 1, 50)

        error = abs(float(result) - expected)
        rel_error = error / expected

        passed = rel_error < 0.01

        self.report.add_result(
            3, "test_integrate_rational", passed,
            "" if passed else f"Rel error: {rel_error:.6f}",
            expected=f"{expected:.6f}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Rational function integration"
        )
        assert passed

    def test_integrate_oscillating(self):
        """Test 3.9: Integrate sin(10x) from 0 to π"""
        f = lambda x: np.sin(10*x)
        # [-cos(10x)/10] from 0 to π = 0
        expected = 0.0

        with TestTimer() as timer:
            result = self.assignment.integrate(f, 0, np.pi, 100)

        error = abs(float(result) - expected)

        passed = error < 0.1

        self.report.add_result(
            3, "test_integrate_oscillating", passed,
            "" if passed else f"Error: {error:.6f}",
            expected=f"{expected}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="High frequency oscillations need many points"
        )
        assert passed

    def test_integrate_small_interval(self):
        """Test 3.10: Integrate on very small interval"""
        f = lambda x: x**2
        # From 0 to 0.001: [x³/3] = (0.001)³/3 = 3.33e-10
        expected = 0.001**3 / 3

        with TestTimer() as timer:
            result = self.assignment.integrate(f, 0, 0.001, 20)

        # Absolute error check for small values
        error = abs(float(result) - expected)

        passed = error < 1e-8

        self.report.add_result(
            3, "test_integrate_small_interval", passed,
            "" if passed else f"Error: {error:.2e}",
            expected=f"{expected:.2e}",
            actual=f"{result:.2e}",
            elapsed_time=timer.elapsed,
            hint="Small intervals need careful handling"
        )
        assert passed

    def test_integrate_negative_result(self):
        """Test 3.11: Integrate function with negative result"""
        f = lambda x: -x**2
        # [-x³/3] from 0 to 2 = -8/3
        expected = -8/3

        with TestTimer() as timer:
            result = self.assignment.integrate(f, 0, 2, 30)

        error = abs(float(result) - expected)

        passed = error < 0.1

        self.report.add_result(
            3, "test_integrate_negative", passed,
            "" if passed else f"Error: {error:.6f}",
            expected=f"{expected:.6f}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Handle negative integrals correctly"
        )
        assert passed

    def test_areabetween_parabolas(self):
        """Test 3.12: Area between two parabolas"""
        f1 = lambda x: x**2
        f2 = lambda x: -x**2 + 4
        # Intersections at x = ±√2
        # Area = integral of (f2 - f1) from -√2 to √2

        with TestTimer() as timer:
            result = self.assignment.areabetween(f1, f2)

        # Expected area ≈ 7.54
        expected = 2 * scipy_integrate.quad(lambda x: 4 - 2*x**2, 0, np.sqrt(2))[0]

        error = abs(float(result) - expected)
        rel_error = error / expected if expected != 0 else error

        passed = rel_error < 0.1

        self.report.add_result(
            3, "test_areabetween_parabolas", passed,
            "" if passed else f"Rel error: {rel_error:.4f}",
            expected=f"{expected:.4f}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Must find intersections and integrate difference"
        )
        assert passed

    def test_areabetween_positive(self):
        """Test 3.13: Area should always be positive"""
        f1 = lambda x: np.sin(x)
        f2 = lambda x: np.cos(x)

        with TestTimer() as timer:
            result = self.assignment.areabetween(f1, f2)

        is_positive = float(result) >= 0

        passed = is_positive

        self.report.add_result(
            3, "test_areabetween_positive", passed,
            "" if passed else f"Area is negative: {result}",
            expected="≥0",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Area cannot be negative"
        )
        assert passed

    def test_areabetween_no_intersection(self):
        """Test 3.14: No intersection should return NaN"""
        f1 = lambda x: x**2 + 5
        f2 = lambda x: x**2

        with TestTimer() as timer:
            result = self.assignment.areabetween(f1, f2)

        is_nan = np.isnan(result)

        passed = is_nan

        self.report.add_result(
            3, "test_areabetween_no_intersection", passed,
            "" if passed else f"Should return NaN, got {result}",
            expected="NaN",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Functions that don't intersect should return NaN"
        )
        assert passed

    def test_float32_precision(self):
        """Test 3.15: Verify all results are float32"""
        f = lambda x: x

        with TestTimer() as timer:
            result = self.assignment.integrate(f, 0, 1, 10)

        is_float32 = isinstance(result, np.float32)

        passed = is_float32

        self.report.add_result(
            3, "test_float32_precision", passed,
            "" if passed else f"Type is {type(result)}, not float32",
            expected="np.float32",
            actual=f"{type(result)}",
            elapsed_time=timer.elapsed,
            hint="All integration results must be float32"
        )
        assert passed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
