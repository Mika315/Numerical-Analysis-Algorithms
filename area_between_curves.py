"""
Finds the area enclosed between two given functions. The rightmost and
leftmost x values for the integration are the rightmost and leftmost
intersection points of the two functions.

This module is more involved than the interpolation and intersection-finding
modules because:
    1. Calculations use float32 precision only (in all calculations), so
       floating point error must be minimized.
    2. There is freedom to choose how to calculate the area between the two
       functions.
    3. The functions may intersect multiple times. Here is an example:
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately.

"""
import numpy as np
from function_intersections import Assignment2

class Assignment3:
    def __init__(self):
        self.ass2 = Assignment2()

        # Cache Gauss-Legendre nodes
        self.gauss_cache = {}
        for n in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
            nodes, weights = np.polynomial.legendre.leggauss(n)
            self.gauss_cache[n] = (
                nodes.astype(np.float32),
                weights.astype(np.float32)
            )

    def _get_gauss_points(self, n):
        if n in self.gauss_cache:
            return self.gauss_cache[n]
        nodes, weights = np.polynomial.legendre.leggauss(n)
        return nodes.astype(np.float32), weights.astype(np.float32)

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
                Integrate the function f in the closed range [a,b] using at most n
                points. The main objective is minimizing the integration error,
                with a secondary objective of minimizing the running time. This is
                tested against a variety of different functions.

                Integration error is measured compared to the actual value of the
                definite integral.

                Note: It is forbidden to call f more than n times.

                Parameters
                ----------
                f : callable. it is the given function
                a : float
                    beginning of the integration range.
                b : float
                    end of the integration range.
                n : int
                    maximal number of points to use.

                Returns
                -------
                np.float32
                    The definite integral of f between a and b
                """
        if n <= 0:
            return np.float32(0.0)

        a = np.float32(a)
        b = np.float32(b)

        # For small n, use single-interval Gauss-Legendre with EXACTLY n points
        # This guarantees we never exceed the function call limit
        if n <= 25:
            return self._gauss_legendre_integrate(f, a, b, n)
        else:
            # For larger n, use composite Gauss-Legendre quadrature
            return self._composite_gauss_legendre(f, a, b, n)

    def _gauss_legendre_integrate(self, f, a, b, n):
        """
        Single-interval Gauss-Legendre quadrature using exactly n points.
        Gauss-Legendre with n points integrates polynomials of degree 2n-1 exactly.
        """
        # Get Gauss-Legendre nodes and weights for exactly n points
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = nodes.astype(np.float32)
        weights = weights.astype(np.float32)

        # Transform from [-1, 1] to [a, b]
        h = (b - a) * np.float32(0.5)
        m = (a + b) * np.float32(0.5)

        # Kahan summation for better float32 accuracy
        total = np.float32(0.0)
        comp = np.float32(0.0)

        for i in range(n):
            x = float(h * nodes[i] + m)
            term = np.float32(f(x)) * weights[i]

            # Kahan summation step
            y = term - comp
            t = total + y
            comp = (t - total) - y
            total = t

        return np.float32(total * h)

    def _composite_gauss_legendre(self, f, a, b, n):
        """
        Composite Gauss-Legendre quadrature for large n.
        Divides [a,b] into subintervals and applies Gauss-Legendre on each.
        CRITICAL: Total function calls must equal exactly n.
        """
        # Find optimal order and interval count such that order * intervals = n
        # Prefer higher order for better accuracy (Gauss-Legendre order k is exact for degree 2k-1)
        best_order = 1
        best_intervals = n

        # Try orders from high to low, pick first that divides n evenly
        for order in [20, 15, 12, 10, 8, 6, 5, 4, 3, 2]:
            if n % order == 0:
                intervals = n // order
                if intervals >= 1:
                    best_order = order
                    best_intervals = intervals
                    break

        # Get Gauss-Legendre nodes/weights for chosen order
        nodes, weights = np.polynomial.legendre.leggauss(best_order)
        nodes = nodes.astype(np.float32)
        weights = weights.astype(np.float32)

        # Compute interval width
        interval_width = (b - a) / np.float32(best_intervals)

        # Kahan summation for numerical stability
        total = np.float32(0.0)
        comp = np.float32(0.0)

        for i in range(best_intervals):
            # Subinterval bounds
            a_i = a + np.float32(i) * interval_width
            b_i = a_i + interval_width

            # Transform parameters for this subinterval
            h = interval_width * np.float32(0.5)
            m = (a_i + b_i) * np.float32(0.5)

            for j in range(best_order):
                x = float(h * nodes[j] + m)
                term = np.float32(f(x)) * weights[j] * h

                # Kahan summation step
                y = term - comp
                t = total + y
                comp = (t - total) - y
                total = t

        return np.float32(total)


    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
                Finds the area enclosed between two functions. This method finds
                all intersection points between the two functions to work correctly.

                Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

                Note, there is no such thing as negative area.

                In order to find the enclosed area the given functions must intersect
                in at least two points. If the functions do not intersect or intersect
                in less than two points this function returns NaN.
                This function may not work correctly if there is infinite number of
                intersection points.


                Parameters
                ----------
                f1,f2 : callable. These are the given functions

                Returns
                -------
                np.float32
                    The area between function and the X axis

                """
        xs = sorted(self.ass2.intersections(f1, f2, 1.0, 100.0, maxerr=0.001))

        if len(xs) < 2:
            return np.float32(np.nan)

        total = np.float32(0.0)
        compensation = np.float32(0.0)

        for i in range(len(xs) - 1):
            x1 = xs[i]
            x2 = xs[i + 1]

            def diff(x):
                return f1(x) - f2(x)

            area = np.abs(self.integrate(diff, x1, x2, 100))

            # Kahan summation
            area_adjusted = area - compensation
            temp = total + area_adjusted
            compensation = (temp - total) - area_adjusted
            total = temp

        return np.float32(total)

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
