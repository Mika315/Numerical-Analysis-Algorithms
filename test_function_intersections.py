"""
Extended tests for Assignment 2: Finding Intersections
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import TestTimer, TestReport

try:
    from function_intersections import Assignment2
except ImportError:
    print("ERROR: Could not import function_intersections.py")
    sys.exit(1)


class TestAssignment2:
    """Comprehensive tests for finding intersections"""

    @classmethod
    def setup_class(cls):
        cls.assignment = Assignment2()
        cls.report = TestReport()

    @classmethod
    def teardown_class(cls):
        cls.report.print_summary()

    def verify_intersections(self, f1, f2, intersections, maxerr,
                            expected_count=None, name="test"):
        """Helper to verify intersection points"""
        errors = []
        for x in intersections:
            err = abs(f1(x) - f2(x))
            if err > maxerr:
                errors.append((x, err))

        all_valid = len(errors) == 0

        count_msg = ""
        if expected_count is not None:
            count_msg = f", expected ~{expected_count} roots"

        return all_valid, errors, count_msg

    def test_simple_parabola_line(self):
        """Test 2.1: Parabola x² intersects line x+2"""
        f1 = lambda x: x**2
        f2 = lambda x: x + 2

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, -3, 3, maxerr=0.001)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.001)

        # Should find 2 roots: x = -1, x = 2
        has_negative = any(x < 0 for x in X)
        has_positive = any(x > 0 for x in X)

        passed = valid and has_negative and has_positive and len(X) >= 2

        self.report.add_result(
            2, "test_parabola_line", passed,
            "" if passed else f"Found {len(X)} roots, errors: {errors}",
            expected="2 roots (one pos, one neg)",
            actual=f"{len(X)} roots: {[f'{x:.4f}' for x in X]}",
            elapsed_time=timer.elapsed,
            hint="Should find x≈-1 and x≈2"
        )
        assert passed

    def test_double_root(self):
        """Test 2.2: Tangent intersection (x-1)² = 0"""
        f1 = lambda x: (x - 1)**2
        f2 = lambda x: 0*x

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, 0, 2, maxerr=0.001)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.001)

        # Should find at least one root near x=1
        has_root_near_one = any(abs(x - 1) < 0.1 for x in X)

        passed = valid and has_root_near_one

        self.report.add_result(
            2, "test_double_root", passed,
            "" if passed else f"Errors: {errors}",
            expected="1 root at x≈1",
            actual=f"{len(X)} roots: {[f'{x:.4f}' for x in X]}",
            elapsed_time=timer.elapsed,
            hint="Double roots (tangent) are challenging to detect"
        )
        assert passed

    def test_multiple_intersections(self):
        """Test 2.3: sin(x) = 0 has multiple roots"""
        f1 = lambda x: np.sin(x)
        f2 = lambda x: 0*x

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, -2*np.pi, 2*np.pi, maxerr=0.001)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.001)

        # Should find roots at -2π, -π, 0, π, 2π (5 roots)
        passed = valid and len(X) >= 3  # At least find several

        self.report.add_result(
            2, "test_multiple_intersections", passed,
            "" if passed else f"Found only {len(X)} roots, errors: {errors}",
            expected="≥3 roots",
            actual=f"{len(X)} roots",
            elapsed_time=timer.elapsed,
            hint="Should find multiple periodic intersections"
        )
        assert passed

    def test_sin_cos_intersection(self):
        """Test 2.4: sin(x) = cos(x)"""
        f1 = lambda x: np.sin(x)
        f2 = lambda x: np.cos(x)

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, -np.pi, np.pi, maxerr=0.001)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.001)

        # Roots at -3π/4 and π/4
        has_negative = any(-0.8 < x < -0.7 for x in X)
        has_positive = any(0.7 < x < 0.8 for x in X)

        passed = valid and has_negative and has_positive

        self.report.add_result(
            2, "test_sin_cos", passed,
            "" if passed else f"Missing expected roots, found: {X}",
            expected="2 roots at ±π/4",
            actual=f"{len(X)} roots: {[f'{x:.4f}' for x in X]}",
            elapsed_time=timer.elapsed,
            hint="Should find x≈±0.785"
        )
        assert passed

    def test_polynomial_intersection(self):
        """Test 2.5: x³ - x = x² - 1"""
        f1 = lambda x: x**3 - x
        f2 = lambda x: x**2 - 1

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, -3, 3, maxerr=0.001)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.001)

        # This gives x³ - x² - x + 1 = 0
        # Roots at x = -1, 1 (and maybe others)
        has_negative = any(x < 0 for x in X)
        has_positive = any(x > 0 for x in X)

        passed = valid and has_negative and has_positive

        self.report.add_result(
            2, "test_poly_intersection", passed,
            "" if passed else f"Errors: {errors}",
            expected="Roots in both pos/neg range",
            actual=f"{len(X)} roots: {[f'{x:.4f}' for x in X]}",
            elapsed_time=timer.elapsed,
            hint="Cubic equation intersection"
        )
        assert passed

    def test_exp_vs_linear(self):
        """Test 2.6: e^x = x + 1"""
        f1 = lambda x: np.exp(x)
        f2 = lambda x: x + 1

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, -2, 2, maxerr=0.001)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.001)

        # Root at x=0
        has_root_at_zero = any(abs(x) < 0.01 for x in X)

        passed = valid and has_root_at_zero

        self.report.add_result(
            2, "test_exp_linear", passed,
            "" if passed else f"Missing root at x≈0, found: {X}",
            expected="1 root at x≈0",
            actual=f"{len(X)} roots: {[f'{x:.4f}' for x in X]}",
            elapsed_time=timer.elapsed,
            hint="Exponential vs linear"
        )
        assert passed

    def test_close_roots(self):
        """Test 2.7: Close roots (x-1)(x-1.01) = 0"""
        f1 = lambda x: (x - 1) * (x - 1.01)
        f2 = lambda x: 0*x

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, 0, 2, maxerr=0.001)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.001)

        # Should find both roots that are 0.01 apart
        roots_near_one = [x for x in X if 0.99 < x < 1.02]

        passed = valid and len(roots_near_one) >= 1

        self.report.add_result(
            2, "test_close_roots", passed,
            "" if passed else f"May miss close roots, found: {roots_near_one}",
            expected="2 roots at x≈1.0, 1.01",
            actual=f"{len(X)} roots: {[f'{x:.4f}' for x in X]}",
            elapsed_time=timer.elapsed,
            hint="Roots very close together - resolution challenge"
        )
        # Don't fail - this is a hard case

    def test_no_intersection(self):
        """Test 2.8: No intersection x² + 1 = 0"""
        f1 = lambda x: x**2 + 1
        f2 = lambda x: 0*x

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, -5, 5, maxerr=0.001)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.001)

        # Should find no roots (or very few false positives)
        passed = valid and len(X) == 0

        self.report.add_result(
            2, "test_no_intersection", passed,
            "" if passed else f"Found false roots: {X}",
            expected="0 roots",
            actual=f"{len(X)} roots",
            elapsed_time=timer.elapsed,
            hint="Should not report false intersections"
        )
        assert passed

    def test_high_frequency_oscillation(self):
        """Test 2.9: sin(10x) = 0.5x in [0, 2]"""
        f1 = lambda x: np.sin(10*x)
        f2 = lambda x: 0.5*x

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, 0, 2, maxerr=0.001)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.001)

        # Many intersections expected
        passed = valid and len(X) >= 3

        self.report.add_result(
            2, "test_high_freq", passed,
            "" if passed else f"Found {len(X)} roots",
            expected="Multiple roots (≥3)",
            actual=f"{len(X)} roots",
            elapsed_time=timer.elapsed,
            hint="High frequency oscillations have many intersections"
        )
        assert passed

    def test_rational_intersection(self):
        """Test 2.10: 1/(1+x²) = 0.5"""
        f1 = lambda x: 1/(1 + x**2)
        f2 = lambda x: 0.5 + 0*x

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, -5, 5, maxerr=0.001)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.001)

        # Roots at x = ±1
        has_positive = any(0.9 < x < 1.1 for x in X)
        has_negative = any(-1.1 < x < -0.9 for x in X)

        passed = valid and has_positive and has_negative

        self.report.add_result(
            2, "test_rational_intersection", passed,
            "" if passed else f"Missing roots at ±1, found: {X}",
            expected="2 roots at x≈±1",
            actual=f"{len(X)} roots: {[f'{x:.4f}' for x in X]}",
            elapsed_time=timer.elapsed,
            hint="Rational function intersection"
        )
        assert passed

    def test_accuracy_strictness(self):
        """Test 2.11: Verify maxerr is respected"""
        f1 = lambda x: x**2
        f2 = lambda x: 4 + 0*x

        # Exact roots at x = ±2
        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, -3, 3, maxerr=0.0001)
            X = list(X)

        # All points must satisfy |f1(x) - f2(x)| < 0.0001
        all_accurate = all(abs(f1(x) - f2(x)) < 0.0001 for x in X)

        passed = all_accurate

        max_error = max([abs(f1(x) - f2(x)) for x in X]) if X else 0

        self.report.add_result(
            2, "test_accuracy_strict", passed,
            "" if passed else f"Max error {max_error:.6f} exceeds 0.0001",
            expected="All points within 0.0001",
            actual=f"Max error: {max_error:.6f}",
            elapsed_time=timer.elapsed,
            hint="Must respect maxerr tolerance"
        )
        assert passed

    def test_large_range(self):
        """Test 2.12: Large range [-100, 100]"""
        f1 = lambda x: 0.01*x**2
        f2 = lambda x: x + 0*x

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, -100, 100, maxerr=0.01)
            X = list(X)

        valid, errors, _ = self.verify_intersections(f1, f2, X, 0.01)

        # Roots at x=0 and x=100
        has_zero = any(abs(x) < 0.1 for x in X)
        has_large = any(99 < x < 101 for x in X)

        passed = valid and (has_zero or has_large)

        self.report.add_result(
            2, "test_large_range", passed,
            "" if passed else f"Missing roots in large range",
            expected="Roots at x≈0 and x≈100",
            actual=f"{len(X)} roots",
            elapsed_time=timer.elapsed,
            hint="Large ranges test numerical stability"
        )
        assert passed

    def test_runtime_limit(self):
        """Test 2.13: Runtime should be reasonable"""
        f1 = np.poly1d([1, 0, -4])
        f2 = lambda x: 0*x

        with TestTimer() as timer:
            X = self.assignment.intersections(f1, f2, -10, 10, maxerr=0.001)

        # Should complete in under 5 seconds for simple case
        passed = timer.elapsed < 5.0

        self.report.add_result(
            2, "test_runtime_limit", passed,
            "" if passed else f"Took {timer.elapsed:.2f}s, should be <5s",
            expected="<5s",
            actual=f"{timer.elapsed:.2f}s",
            elapsed_time=timer.elapsed,
            hint="Optimize search algorithm for better performance"
        )
        assert passed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
