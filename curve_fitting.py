"""
Fits a model function to data sampled from a given (noisy) function.

The sampled data is very noisy, so the goal is to minimize the mean least
squares error between the fitted model and the sampled data points.

The fitting function respects a running-time budget: it receives the
maximum allowed running time as an argument, and is guaranteed to return
at most 5 seconds after that time elapses. An iterative approach should
break out of its optimization loops ahead of time if it knows a single
iteration may take more than 1-2 seconds.

Note: No external numeric optimization libraries are used for this module;
fitting is implemented from scratch via a custom least-squares solver
(see _gaussian_engine below).

"""

import numpy as np
import time
import random

class Assignment4:
    def __init__(self):
        # numerical stability threshold
        self.precision_limit = 1e-15

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        # start timing
        start_clock = time.time()
        stop_margin = 0.5 if maxtime > 1.0 else 0.2 * maxtime

        # buffers for sampled points
        x_buf, y_buf = [], []
        sample_limit = 10000
        count = 0
        n_cheb = min(3 * (d + 1), 100)

        # try sampling interval edges
        for edge in (a, b):
            try:
                val = f(edge)
            except Exception:
                continue
            x_buf.append(edge)
            y_buf.append(val)
            count += 1

        # generate Chebyshev nodes
        indices = np.arange(n_cheb)
        nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(
            np.pi * (2 * indices + 1) / (2 * n_cheb)
        )

        # evaluate function on Chebyshev nodes
        for n_x in nodes:
            if time.time() - start_clock > maxtime - stop_margin:
                break
            try:
                n_y = f(n_x)
            except Exception:
                continue
            x_buf.append(n_x)
            y_buf.append(n_y)
            count += 1

        # random sampling until limit or time runs out
        while count < sample_limit and (time.time() - start_clock) < (maxtime - stop_margin):
            rx = random.uniform(a, b)
            try:
                ry = f(rx)
            except Exception:
                continue
            x_buf.append(rx)
            y_buf.append(ry)
            count += 1

        # fallback if no samples collected
        if len(x_buf) == 0:
            return lambda t: 0.0

        try:
            # convert to numpy arrays
            vec_x = np.asarray(x_buf)
            vec_y = np.asarray(y_buf)

            # build Vandermonde-like matrix
            powers = np.arange(d, -1, -1)
            mat_a = vec_x[:, None] ** powers

            # normal equations components
            gram_m = mat_a.T.dot(mat_a)
            rhs_v = mat_a.T.dot(vec_y)

            # solve linear system
            params = self._gaussian_engine(gram_m, rhs_v)
            return np.poly1d(params)

        except Exception:
            # return mean value if fitting fails
            fallback = np.mean(y_buf) if y_buf else 0.0
            return lambda t: fallback

    def _gaussian_engine(self, a_matrix, b_vector):
        # copy inputs to avoid modification
        mtx = a_matrix.astype(float).copy()
        vec = b_vector.astype(float).copy()
        size = vec.shape[0]

        # forward elimination with partial pivoting
        for i in range(size):
            piv = i + np.argmax(np.abs(mtx[i:, i]))
            if piv != i:
                mtx[[i, piv]] = mtx[[piv, i]]
                vec[[i, piv]] = vec[[piv, i]]

            diag = mtx[i, i]
            if abs(diag) < self.precision_limit:
                continue

            inv_diag = 1.0 / diag
            mtx[i] *= inv_diag
            vec[i] *= inv_diag

            for row in range(i + 1, size):
                factor = mtx[row, i]
                if factor != 0.0:
                    mtx[row] -= factor * mtx[i]
                    vec[row] -= factor * vec[i]

        # back substitution
        res = np.zeros_like(vec)
        for i in range(size - 1, -1, -1):
            res[i] = vec[i] - np.dot(mtx[i, i + 1:], res[i + 1:])

        return res



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEqual(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
