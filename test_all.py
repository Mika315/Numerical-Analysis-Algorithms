"""
Master test runner for all assignments
Run with: python -m pytest test_all.py -v
"""

import pytest
import sys
import os

# Import all test modules
from test_interpolation import TestAssignment1
from test_function_intersections import TestAssignment2
from test_area_between_curves import TestAssignment3
from test_curve_fitting import TestAssignment4
from test_shape_fitting import TestAssignment5


def main():
    """Run all tests and generate comprehensive report"""
    print("="*80)
    print("NUMERICAL ANALYSIS PROJECT - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nRunning extended tests for all 5 assignments...")
    print("This will test:")
    print("  - Task 1: Function Interpolation")
    print("  - Task 2: Finding Intersections")
    print("  - Task 3: Integration and Area Between Curves")
    print("  - Task 4: Noisy Function Fitting")
    print("  - Task 5: Shape Area and Fitting")
    print("\n" + "="*80 + "\n")

    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "-s"  # Show print statements
    ])

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
