import numpy as np
from collections.abc import Iterable
"""
Finds the intersection points of two given functions.
"""

class Assignment2:
    def __init__(self):
        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. This is tested on
        functions that have at least two intersection points, one with a
        positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.
        """

        # Define the difference function g(x) = f1(x) - f2(x)
        # We look for roots of g(x) = 0.
        def g(x):
            return f1(x) - f2(x)

        roots = []
        # Divide range [a, b] into small segments to scan for potential intersections.
        # 2000 segments ensures we catch high-frequency oscillations like f3 or f11.
        n_segments = 2000
        x_points = np.linspace(a, b, n_segments)

        # Iterating through intervals to locate roots
        for i in range(len(x_points) - 1):
            x0, x1 = x_points[i], x_points[i + 1]
            y0, y1 = g(x0), g(x1)

            # Check for a sign change (Bolzano's theorem)
            if y0 * y1 <= 0:
                # Solve using hybrid logic (Secant first, then Bisection)
                root = self._solve_hybrid(g, x0, x1, y0, y1, maxerr)
                if root is not None:
                    # Filter out duplicate points that fall within the error margin
                    if not any(abs(root - r) < maxerr * 10 for r in roots):
                        roots.append(root)

            # Edge case: Functions might touch (tangent) without crossing signs
            elif abs(y0) < maxerr:
                if not any(abs(x0 - r) < maxerr * 10 for r in roots):
                    roots.append(x0)

        return roots

    def _solve_hybrid(self, g, a, b, fa, fb, maxerr):
        """
        Hybrid solver: Try Secant for speed; use Bisection if Secant fails or diverges.
        """
        # --- Fast Path: Secant Method ---
        x_pre, x_curr = a, b
        f_pre, f_curr = fa, fb

        for _ in range(12):  # Limit iterations to maintain speed cap.
            if abs(f_curr - f_pre) < 1e-15:
                break  # Avoid division by zero

            # Linear interpolation (Secant step)
            x_next = x_curr - f_curr * (x_curr - x_pre) / (f_curr - f_pre)

            # Safety: Ensure the step stays within the current sub-interval
            if not (min(a, b) <= x_next <= max(a, b)):
                break

            f_next = g(x_next)
            if abs(f_next) < maxerr:
                return x_next

            x_pre, x_curr = x_curr, x_next
            f_pre, f_curr = f_curr, f_next

        # --- Robust Path: Bisection Method ---
        # Guaranteed to find a root if fa * fb <= 0.
        low, high = a, b
        f_low, f_high = fa, fb

        if f_low * f_high > 0:
            return None  # No guaranteed crossing here

        for _ in range(64):  # High iterations for float64 precision
            mid = (low + high) / 2
            f_mid = g(mid)
            if abs(f_mid) < maxerr:
                return mid

            if f_low * f_mid < 0:
                high, f_high = mid, f_mid
            else:
                low, f_low = mid, f_mid

        return (low + high) / 2
##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
