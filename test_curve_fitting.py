"""
Extended tests for Assignment 4: Noisy Function Fitting
Includes hard cases and official grader-style reporting.
"""

import pytest
import numpy as np
import sys
import os
import time

# Path setup for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import TestTimer, TestReport
# Import new helper functions added to utils
try:
    from utils import exit_after, calculate_relative_error, hard_functions_generator, add_outlier_noise
except ImportError:
    pass

try:
    from curve_fitting import Assignment4
except ImportError:
    print("ERROR: Could not import curve_fitting.py")
    sys.exit(1)


def add_noise(f, noise_level=0.1):
    """Add Gaussian noise to function"""
    def noisy_f(x):
        return f(x) + np.random.normal(0, noise_level)
    return noisy_f


class TestAssignment4:
    """Comprehensive tests for noisy function fitting"""

    @classmethod
    def setup_class(cls):
        cls.assignment = Assignment4()
        cls.report = TestReport()
        np.random.seed(42)  # For reproducible noise

    @classmethod
    def teardown_class(cls):
        cls.report.print_summary()

    # --- Basic tests (4.1 - 4.12) ---

    def test_fit_constant(self):
        """Test 4.1: Fit constant function with noise"""
        f_clean = lambda x: 5.0
        f_noisy = add_noise(f_clean, 0.5)
        with TestTimer() as timer:
            f_fitted = self.assignment.fit(f_noisy, 0, 10, d=0, maxtime=5)
        test_points = np.linspace(0, 10, 50)
        errors = [abs(f_fitted(x) - f_clean(x)) for x in test_points]
        mean_error = np.mean(errors)
        passed = mean_error < 1.0 and timer.elapsed <= 7
        self.report.add_result(4, "test_fit_constant", passed, actual=f"{mean_error:.4f} error", elapsed_time=timer.elapsed)
        assert passed

    def test_fit_linear(self):
        """Test 4.2: Fit linear function y=2x+3"""
        f_clean = lambda x: 2*x + 3
        f_noisy = add_noise(f_clean, 0.5)
        with TestTimer() as timer:
            f_fitted = self.assignment.fit(f_noisy, -5, 5, d=1, maxtime=5)
        errors = [abs(f_fitted(x) - f_clean(x)) for x in np.linspace(-5, 5, 50)]
        mean_error = np.mean(errors)
        passed = mean_error < 1.0
        self.report.add_result(4, "test_fit_linear", passed, actual=f"{mean_error:.4f}", elapsed_time=timer.elapsed)
        assert passed

    def test_fit_quadratic(self):
        """Test 4.3: Fit quadratic"""
        f_clean = np.poly1d([1, -3, 5])
        f_noisy = add_noise(f_clean, 1.0)
        with TestTimer() as timer:
            f_fitted = self.assignment.fit(f_noisy, -5, 5, d=2, maxtime=5)
        mean_error = np.mean([abs(f_fitted(x) - f_clean(x)) for x in np.linspace(-5, 5, 50)])
        passed = mean_error < 2.0
        self.report.add_result(4, "test_fit_quadratic", passed, actual=f"{mean_error:.4f}", elapsed_time=timer.elapsed)
        assert passed

    def test_fit_cubic(self):
        """Test 4.4: Fit cubic polynomial"""
        f_clean = np.poly1d([1, -2, 1, -1])
        f_noisy = add_noise(f_clean, 0.8)
        with TestTimer() as timer:
            f_fitted = self.assignment.fit(f_noisy, -3, 3, d=3, maxtime=5)
        mean_error = np.mean([abs(f_fitted(x) - f_clean(x)) for x in np.linspace(-3, 3, 50)])
        passed = mean_error < 2.0
        self.report.add_result(4, "test_fit_cubic", passed, actual=f"{mean_error:.4f}", elapsed_time=timer.elapsed)
        assert passed

    def test_fit_high_degree(self):
        """Test 4.5: Fit degree 8 polynomial"""
        coeffs = np.random.uniform(-0.3, 0.3, 9)
        f_clean = np.poly1d(coeffs)
        f_noisy = add_noise(f_clean, 1.0)
        with TestTimer() as timer:
            f_fitted = self.assignment.fit(f_noisy, -2, 2, d=8, maxtime=8)
        mean_error = np.mean([abs(f_fitted(x) - f_clean(x)) for x in np.linspace(-2, 2, 100)])
        passed = mean_error < 5.0
        self.report.add_result(4, "test_fit_high_degree", passed, actual=f"{mean_error:.4f}", elapsed_time=timer.elapsed)
        assert passed

    def test_time_constraint_respected(self):
        """Test 4.6: Must respect maxtime constraint"""
        f_noisy = add_noise(lambda x: x**2, 0.5)
        maxtime = 2.0
        with TestTimer() as timer:
            f_fitted = self.assignment.fit(f_noisy, 0, 10, d=2, maxtime=maxtime)
        passed = timer.elapsed <= maxtime + 5
        self.report.add_result(4, "test_time_constraint", passed, actual=f"{timer.elapsed:.2f}s", elapsed_time=timer.elapsed)
        assert passed

    def test_high_noise_robustness(self):
        """Test 4.7: Handle high noise levels"""
        f_clean = lambda x: 3*x + 2
        f_noisy = add_noise(f_clean, 5.0)
        with TestTimer() as timer:
            f_fitted = self.assignment.fit(f_noisy, -10, 10, d=1, maxtime=5)
        mean_error = np.mean([abs(f_fitted(x) - f_clean(x)) for x in np.linspace(-10, 10, 50)])
        passed = mean_error < 10.0
        self.report.add_result(4, "test_high_noise", passed, actual=f"{mean_error:.4f}", elapsed_time=timer.elapsed)
        assert passed

    def test_non_polynomial_hint(self):
        """Test 4.8: Non-polynomial (exponential)"""
        f_clean = lambda x: np.exp(x/5)
        f_noisy = add_noise(f_clean, 0.1)
        with TestTimer() as timer:
            f_fitted = self.assignment.fit(f_noisy, 0, 5, d=3, maxtime=5)
        mean_error = np.mean([abs(f_fitted(x) - f_clean(x)) for x in np.linspace(0, 5, 50)])
        passed = mean_error < 1.0
        self.report.add_result(4, "test_non_polynomial", passed, actual=f"{mean_error:.4f}", elapsed_time=timer.elapsed)

    def test_varying_sample_sizes(self):
        """Test 4.9: More samples for better accuracy"""
        f_clean = lambda x: x**2
        f_noisy = add_noise(f_clean, 1.0)
        with TestTimer() as timer:
            f_fitted = self.assignment.fit(f_noisy, -5, 5, d=2, maxtime=5)
        mean_error = np.mean([abs(f_fitted(x) - f_clean(x)) for x in np.linspace(-5, 5, 50)])
        passed = mean_error < 3.0
        self.report.add_result(4, "test_sample_size", passed, actual=f"{mean_error:.4f}", elapsed_time=timer.elapsed)
        assert passed

    def test_edge_case_single_point_range(self):
        """Test 4.10: Very small range"""
        f_clean = lambda x: x + 1
        f_noisy = add_noise(f_clean, 0.1)
        f_fitted = self.assignment.fit(f_noisy, 0, 0.1, d=1, maxtime=3)
        mean_error = np.mean([abs(f_fitted(x) - f_clean(x)) for x in np.linspace(0, 0.1, 20)])
        assert mean_error < 0.5

    def test_negative_range(self):
        """Test 4.11: Negative range [-10, -5]"""
        f_clean = lambda x: -x**2 + 3*x
        f_noisy = add_noise(f_clean, 0.5)
        f_fitted = self.assignment.fit(f_noisy, -10, -5, d=2, maxtime=5)
        mean_error = np.mean([abs(f_fitted(x) - f_clean(x)) for x in np.linspace(-10, -5, 30)])
        assert mean_error < 2.0

    def test_degree_mismatch(self):
        """Test 4.12: Degree mismatch"""
        f_clean = np.poly1d([1, 0, -4])
        f_noisy = add_noise(f_clean, 0.8)
        f_fitted = self.assignment.fit(f_noisy, -3, 3, d=5, maxtime=5)
        mean_error = np.mean([abs(f_fitted(x) - f_clean(x)) for x in np.linspace(-3, 3, 50)])
        assert mean_error < 3.0

    # --- Advanced tests and edge cases (Grader style) ---

    def test_hard_cases_grader_style(self):
        """Test 4.13: Official Grader Style with Hard Functions"""
        hard_cases = hard_functions_generator()
        print("\n" + "="*50)
        print("RUNNING GRADER-STYLE HARD CASES")
        print("="*50)

        for f_true, a, b, d, name in hard_cases:
            f_noisy = add_noise(f_true, noise_level=0.1)
            maxtime = 5.0

            @exit_after(maxtime + 2)  # Handle timeout
            def run_fit():
                return self.assignment.fit(f_noisy, a, b, d, maxtime)

            start_t = time.time()
            try:
                f_fitted = run_fit()
                elapsed = time.time() - start_t
                rel_err = calculate_relative_error(f_fitted, f_true, a, b)

                # Print format: RES | EXP | ERR
                print(f"RES: {rel_err:.6f} | EXP: 0.000000 | ERR: {rel_err:.6f} | Time: {elapsed:.2f}s | Task: {name}")

                passed = rel_err < 0.5
                self.report.add_result(4, f"hard_{name}", passed, actual=f"RelErr: {rel_err:.4%}", elapsed_time=elapsed)
            except Exception as e:
                print(f"ERR: {name} failed with {e}")
                self.report.add_result(4, f"hard_{name}", False, error_msg=str(e))

    def test_outlier_robustness(self):
        """Test 4.14: Robustness against extreme outliers"""
        f_clean = lambda x: 0.5 * x**2 + 2
        f_noisy = add_outlier_noise(f_clean, noise_level=0.5, outlier_prob=0.1)

        with TestTimer() as timer:
            f_fitted = self.assignment.fit(f_noisy, -5, 5, d=2, maxtime=7)

        rel_err = calculate_relative_error(f_fitted, f_clean, -5, 5)
        print(f"Outlier Test: RES: {rel_err:.6f} | ERR: {rel_err:.6f} | Time: {timer.elapsed:.2f}s")

        passed = rel_err < 0.6
        self.report.add_result(4, "test_outliers", passed, actual=f"Rel: {rel_err:.2%}", elapsed_time=timer.elapsed)
        assert passed

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
