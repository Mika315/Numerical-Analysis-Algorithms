import numpy as np
import time
import random
"""
Function interpolation: builds an approximating function for a given
black-box function.
"""


class Assignment1:
    def __init__(self):
        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. The main objective is minimizing the interpolation error,
        with a secondary objective of minimizing the running time. This is
        tested against a variety of different functions with large n values.

        Interpolation error is measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This can be solved trivially with running time O(n^2), or with
        running time O(n) using some preprocessing.

        Note: sometimes very accurate solutions can be found with only a few
        points, significantly less than n.
        n<=100
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        if n == 1:
            y_val = f(a)
            return lambda x: y_val

            # Use equidistant nodes for O(1) index lookup
        x_samples = np.linspace(a, b, n)
        h = (b - a) / (n - 1)

        # Guard against division by very small h
        if h < 1e-14:
            y_val = f(a)
            return lambda x: y_val

        inv_h = 1.0 / h

        # Evaluate function exactly n times
        y_samples = np.array([f(xi) for xi in x_samples], dtype=np.float64)

        # Build cubic spline using Thomas algorithm
        spline_data = self._build_cubic_spline(y_samples, h, n)

        def interpolation_function(x):
            # Fast scalar path
            if isinstance(x, (int, float, np.number)):
                if x <= a:
                    return y_samples[0]
                if x >= b:
                    return y_samples[-1]

                idx = int((x - a) * inv_h)
                if idx >= n - 1:
                    idx = n - 2

                t = (x - x_samples[idx]) * inv_h
                coeffs = spline_data[idx]

                # Horner's method for numerical stability
                return coeffs[0] + t * (coeffs[1] + t * (coeffs[2] + t * coeffs[3]))

            # Vectorized path
            x_arr = np.clip(np.atleast_1d(x), a, b)
            indices = ((x_arr - a) * inv_h).astype(int)
            indices = np.clip(indices, 0, n - 2)

            t = (x_arr - x_samples[indices]) * inv_h
            c = spline_data[indices]
            return c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * c[:, 3]))

        return interpolation_function

    def _build_cubic_spline(self, y, h, n):
        """
        Natural cubic spline via Thomas algorithm - O(n) complexity.
        """
        if n == 2:
            return np.array([[y[0], y[1] - y[0], 0.0, 0.0]])

        num_internal = n - 2
        diag = np.ones(num_internal) * 4.0
        h2 = h * h
        rhs = 6.0 * (y[2:] - 2 * y[1:-1] + y[:-2]) / h2

        # Thomas forward sweep
        cp = np.zeros(num_internal)
        dp = np.zeros(num_internal)
        cp[0] = 1.0 / diag[0]
        dp[0] = rhs[0] / diag[0]

        for i in range(1, num_internal):
            m = diag[i] - cp[i - 1]
            cp[i] = 1.0 / m
            dp[i] = (rhs[i] - dp[i - 1]) / m

        # Thomas backward sweep
        m_vals = np.zeros(n)
        m_vals[n - 2] = dp[-1]
        for i in range(num_internal - 2, -1, -1):
            m_vals[i + 1] = dp[i] - cp[i] * m_vals[i + 2]

        # Compute polynomial coefficients
        coeffs = np.zeros((n - 1, 4))
        for i in range(n - 1):
            A = y[i]
            C = m_vals[i] * h2 / 2.0
            D = (m_vals[i + 1] - m_vals[i]) * h2 / 6.0
            B = (y[i + 1] - y[i]) - (C + D)
            coeffs[i] = [A, B, C, D]

        return coeffs
##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm




class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
