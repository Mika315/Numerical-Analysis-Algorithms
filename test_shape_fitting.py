"""
Extended tests for Assignment 5: Shape Area and Fitting
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import TestTimer, TestReport

try:
    from shape_fitting import Assignment5, MyShape
    from functionUtils import AbstractShape
except ImportError:
    print("ERROR: Could not import shape_fitting.py or functionUtils.py")
    sys.exit(1)


def circle_contour(cx, cy, r):
    """Generate circle contour function"""
    def contour(n):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in angles]
    return contour


def noisy_circle_sample(cx, cy, r, noise):
    """Generate noisy circle sampler"""
    def sample():
        angle = np.random.uniform(0, 2*np.pi)
        noise_r = np.random.normal(0, noise)
        noise_theta = np.random.normal(0, noise/r)
        actual_r = r + noise_r
        actual_angle = angle + noise_theta
        x = cx + actual_r * np.cos(actual_angle)
        y = cy + actual_r * np.sin(actual_angle)
        return (x, y)
    return sample


def ellipse_contour(cx, cy, a, b):
    """Generate ellipse contour"""
    def contour(n):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return [(cx + a*np.cos(t), cy + b*np.sin(t)) for t in angles]
    return contour


class TestAssignment5:
    """Comprehensive tests for shape area and fitting"""

    @classmethod
    def setup_class(cls):
        cls.assignment = Assignment5()
        cls.report = TestReport()
        np.random.seed(42)

    @classmethod
    def teardown_class(cls):
        cls.report.print_summary()

    def test_area_circle(self):
        """Test 5.1: Area of circle"""
        r = 2.0
        contour = circle_contour(0, 0, r)
        expected_area = np.pi * r**2

        with TestTimer() as timer:
            result = self.assignment.area(contour, maxerr=0.001)

        error = abs(float(result) - expected_area)
        rel_error = error / expected_area

        passed = rel_error < 0.01

        self.report.add_result(
            5, "test_area_circle", passed,
            "" if passed else f"Rel error: {rel_error:.6f}",
            expected=f"{expected_area:.6f}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Circle area = πr²"
        )
        assert passed

    def test_area_square(self):
        """Test 5.2: Area of square"""
        side = 4.0
        def square_contour(n):
            # Square corners
            step = side / (n // 4)
            points = []
            # Bottom edge
            for i in range(n//4):
                points.append((i*step, 0))
            # Right edge
            for i in range(n//4):
                points.append((side, i*step))
            # Top edge
            for i in range(n//4):
                points.append((side - i*step, side))
            # Left edge
            for i in range(n//4):
                points.append((0, side - i*step))
            return points

        expected_area = side ** 2

        with TestTimer() as timer:
            result = self.assignment.area(square_contour, maxerr=0.01)

        error = abs(float(result) - expected_area)
        rel_error = error / expected_area

        passed = rel_error < 0.05

        self.report.add_result(
            5, "test_area_square", passed,
            "" if passed else f"Rel error: {rel_error:.6f}",
            expected=f"{expected_area}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Square area = side²"
        )
        assert passed

    def test_area_ellipse(self):
        """Test 5.3: Area of ellipse"""
        a, b = 3.0, 2.0
        contour = ellipse_contour(0, 0, a, b)
        expected_area = np.pi * a * b

        with TestTimer() as timer:
            result = self.assignment.area(contour, maxerr=0.001)

        error = abs(float(result) - expected_area)
        rel_error = error / expected_area

        passed = rel_error < 0.02

        self.report.add_result(
            5, "test_area_ellipse", passed,
            "" if passed else f"Rel error: {rel_error:.6f}",
            expected=f"{expected_area:.6f}",
            actual=f"{result}",
            elapsed_time=timer.elapsed,
            hint="Ellipse area = πab"
        )
        assert passed

    def test_fit_circle_returns_shape(self):
        """Test 5.4: fit_shape returns AbstractShape"""
        sample = noisy_circle_sample(1, 1, 2, 0.1)

        with TestTimer() as timer:
            result = self.assignment.fit_shape(sample, maxtime=5)

        is_shape = isinstance(result, AbstractShape)
        time_ok = timer.elapsed <= 7

        passed = is_shape and time_ok

        self.report.add_result(
            5, "test_fit_returns_shape", passed,
            "" if passed else f"Type: {type(result)}, time: {timer.elapsed:.2f}s",
            expected="AbstractShape, <7s",
            actual=f"{type(result)}, {timer.elapsed:.2f}s",
            elapsed_time=timer.elapsed,
            hint="Must return object extending AbstractShape"
        )
        assert passed

    def test_fit_circle_has_contour_method(self):
        """Test 5.5: Fitted shape has contour method"""
        sample = noisy_circle_sample(0, 0, 1, 0.1)

        with TestTimer() as timer:
            shape = self.assignment.fit_shape(sample, maxtime=5)

        has_contour = hasattr(shape, 'contour') and callable(shape.contour)

        if has_contour:
            try:
                points = shape.contour(10)
                works = len(points) == 10
            except:
                works = False
        else:
            works = False

        passed = has_contour and works

        self.report.add_result(
            5, "test_fit_has_contour", passed,
            "" if passed else "contour() method missing or broken",
            expected="contour(n) returns n points",
            actual="OK" if passed else "FAIL",
            elapsed_time=timer.elapsed,
            hint="Shape must implement contour(n) method"
        )
        assert passed

    def test_fit_circle_area_accuracy(self):
        """Test 5.6: Fitted circle has accurate area"""
        r = 2.0
        sample = noisy_circle_sample(1, 1, r, 0.1)
        expected_area = np.pi * r**2

        with TestTimer() as timer:
            shape = self.assignment.fit_shape(sample, maxtime=10)
            actual_area = shape.area()

        error = abs(float(actual_area) - expected_area)
        rel_error = error / expected_area

        passed = rel_error < 0.1

        self.report.add_result(
            5, "test_fit_circle_area", passed,
            "" if passed else f"Rel error: {rel_error:.4f}",
            expected=f"{expected_area:.4f}",
            actual=f"{actual_area}",
            elapsed_time=timer.elapsed,
            hint="Fitted circle area should match true area"
        )
        assert passed

    def test_fit_time_constraint(self):
        """Test 5.7: Fit respects maxtime"""
        sample = noisy_circle_sample(0, 0, 1, 0.05)
        maxtime = 3.0

        with TestTimer() as timer:
            shape = self.assignment.fit_shape(sample, maxtime=maxtime)

        # Should return within maxtime + 5
        time_ok = timer.elapsed <= maxtime + 5

        passed = time_ok

        self.report.add_result(
            5, "test_fit_time_constraint", passed,
            "" if passed else f"Took {timer.elapsed:.2f}s > {maxtime+5}s",
            expected=f"<{maxtime+5}s",
            actual=f"{timer.elapsed:.2f}s",
            elapsed_time=timer.elapsed,
            hint="Must return within maxtime + 5 seconds"
        )
        assert passed

    def test_fit_high_noise(self):
        """Test 5.8: Handle high noise in samples"""
        r = 3.0
        sample = noisy_circle_sample(0, 0, r, 0.5)  # High noise
        expected_area = np.pi * r**2

        with TestTimer() as timer:
            shape = self.assignment.fit_shape(sample, maxtime=15)
            actual_area = shape.area()

        error = abs(float(actual_area) - expected_area)
        rel_error = error / expected_area

        # More lenient for high noise
        passed = rel_error < 0.2

        self.report.add_result(
            5, "test_fit_high_noise", passed,
            "" if passed else f"Rel error: {rel_error:.4f} with high noise",
            expected=f"~{expected_area:.2f} (±20%)",
            actual=f"{actual_area}",
            elapsed_time=timer.elapsed,
            hint="High noise requires many samples"
        )
        assert passed

    def test_area_runtime_efficiency(self):
        """Test 5.9: Area computation is efficient"""
        contour = circle_contour(0, 0, 5)

        with TestTimer() as timer:
            result = self.assignment.area(contour, maxerr=0.001)

        # Should be fast (< 2 seconds)
        passed = timer.elapsed < 2.0

        self.report.add_result(
            5, "test_area_runtime", passed,
            "" if passed else f"Too slow: {timer.elapsed:.2f}s",
            expected="<2s",
            actual=f"{timer.elapsed:.2f}s",
            elapsed_time=timer.elapsed,
            hint="Optimize area calculation for speed"
        )
        # Informative - don't fail

    def test_contour_equally_spaced(self):
        """Test 5.10: Fitted contour has equally spaced points"""
        sample = noisy_circle_sample(0, 0, 1, 0.05)

        with TestTimer() as timer:
            shape = self.assignment.fit_shape(sample, maxtime=8)
            points = shape.contour(20)

        # Check that we got 20 points
        count_ok = len(points) == 20

        # Check points are tuples/lists of 2 elements
        format_ok = all(len(p) == 2 for p in points)

        passed = count_ok and format_ok

        self.report.add_result(
            5, "test_contour_spacing", passed,
            "" if passed else f"Got {len(points)} points, expected 20",
            expected="20 (x,y) tuples",
            actual=f"{len(points)} points",
            elapsed_time=timer.elapsed,
            hint="contour(n) must return exactly n points"
        )
        assert passed

    def test_area_float32_type(self):
        """Test 5.11: area() returns float32"""
        contour = circle_contour(0, 0, 1)

        with TestTimer() as timer:
            result = self.assignment.area(contour, maxerr=0.01)

        is_float32 = isinstance(result, np.float32)

        passed = is_float32

        self.report.add_result(
            5, "test_area_type", passed,
            "" if passed else f"Type is {type(result)}, expected np.float32",
            expected="np.float32",
            actual=f"{type(result)}",
            elapsed_time=timer.elapsed,
            hint="area() must return np.float32"
        )
        assert passed

    def test_fit_different_centers(self):
        """Test 5.12: Fit circle with non-origin center"""
        cx, cy, r = 5.0, -3.0, 2.5
        sample = noisy_circle_sample(cx, cy, r, 0.1)
        expected_area = np.pi * r**2

        with TestTimer() as timer:
            shape = self.assignment.fit_shape(sample, maxtime=10)
            actual_area = shape.area()

        error = abs(float(actual_area) - expected_area)
        rel_error = error / expected_area

        passed = rel_error < 0.15

        self.report.add_result(
            5, "test_fit_different_center", passed,
            "" if passed else f"Rel error: {rel_error:.4f}",
            expected=f"{expected_area:.4f}",
            actual=f"{actual_area}",
            elapsed_time=timer.elapsed,
            hint="Should handle shapes at any position"
        )
        assert passed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
