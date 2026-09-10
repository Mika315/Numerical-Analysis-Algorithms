"""
Fits a model function to data sampled from the contour of a given shape,
then calculates the area of that shape.

The sampled data is very noisy, so the goal is to minimize the mean least
squares error between the fitted model and the sampled data points.

The fitting function respects a running-time budget: it receives the
maximum allowed running time as an argument, and is guaranteed to return
at most 5 seconds after that time elapses. An iterative approach should
break out of its optimization loops ahead of time if it knows a single
iteration may take more than 1-2 seconds.

Note: numeric optimization libraries and tools may be used for this module.
Note: only (x, y) points obtained via `sample()` may be used — inspecting
the sampled function's internal parameters directly (e.g. via reflection)
is not permitted, since the fitting must work from observed data alone.
"""

import numpy as np
import time
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    """
    Shape representation using Fourier descriptors.
    Robust to noise and handles arbitrary closed curves.
    """

    def __init__(self, coeffs_x=None, coeffs_y=None, n_harmonics=20):
        """
        Initialize shape with Fourier coefficients.
        x(t) = a0 + sum(a_k*cos(k*t) + b_k*sin(k*t))
        y(t) = c0 + sum(c_k*cos(k*t) + d_k*sin(k*t))

        Coefficients stored as: [a0, a1, b1, a2, b2, ...]
        """
        self.n_harmonics = n_harmonics

        if coeffs_x is not None:
            self.coeffs_x = np.asarray(coeffs_x, dtype=np.float64)
            self.coeffs_y = np.asarray(coeffs_y, dtype=np.float64)
        else:
            # Default: unit circle
            self.coeffs_x = np.zeros(2 * n_harmonics + 1, dtype=np.float64)
            self.coeffs_y = np.zeros(2 * n_harmonics + 1, dtype=np.float64)
            self.coeffs_x[1] = 1.0  # a1 = 1 (cos term for x)
            self.coeffs_y[2] = 1.0  # b1 = 1 (sin term for y)

    def _eval_fourier(self, t, coeffs):
        """Evaluate Fourier series at parameter t (vectorized)."""
        t = np.atleast_1d(t)
        result = np.full_like(t, coeffs[0], dtype=np.float64)

        for k in range(1, self.n_harmonics + 1):
            a_idx = 2 * k - 1
            b_idx = 2 * k
            if a_idx < len(coeffs):
                result += coeffs[a_idx] * np.cos(k * t)
            if b_idx < len(coeffs):
                result += coeffs[b_idx] * np.sin(k * t)

        return result

    def contour(self, n: int):
        """Return n equally spaced points on the contour."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = self._eval_fourier(t, self.coeffs_x)
        y = self._eval_fourier(t, self.coeffs_y)
        return np.column_stack([x, y])

    def area(self) -> np.float32:
        """
        Calculate area using analytical formula from Fourier coefficients.

        Green's theorem: Area = 0.5 * |∮ x dy - y dx|

        For Fourier series, this simplifies to:
        Area = π * |Σ k * (a_k * d_k - b_k * c_k)|

        where x = a0 + Σ(a_k*cos(kt) + b_k*sin(kt))
              y = c0 + Σ(c_k*cos(kt) + d_k*sin(kt))
        """
        area = 0.0

        for k in range(1, self.n_harmonics + 1):
            a_idx = 2 * k - 1  # cos coefficient index
            b_idx = 2 * k      # sin coefficient index

            if b_idx >= len(self.coeffs_x) or b_idx >= len(self.coeffs_y):
                break

            a_k = self.coeffs_x[a_idx]  # cos coeff for x
            b_k = self.coeffs_x[b_idx]  # sin coeff for x
            c_k = self.coeffs_y[a_idx]  # cos coeff for y
            d_k = self.coeffs_y[b_idx]  # sin coeff for y

            # Green's theorem contribution
            area += k * (a_k * d_k - b_k * c_k)

        area = np.pi * abs(area)

        # Numerical fallback if analytical result seems invalid
        if area < 1e-10 or not np.isfinite(area):
            area = self._numerical_area()

        return np.float32(area)

    def _numerical_area(self):
        """Shoelace formula on dense contour (fallback)."""
        pts = self.contour(5000)
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class Assignment5:
    def __init__(self):
        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.
        Uses adaptive refinement with Richardson extrapolation.
        """
        n = 64
        pts1 = contour(n)
        A1 = self._shoelace(pts1)

        # Richardson extrapolation for faster convergence
        for _ in range(12):
            n2 = n * 2
            if n2 > 10000:
                break

            pts2 = contour(n2)
            A2 = self._shoelace(pts2)

            # Extrapolate (polygon area error ~ O(h^2))
            A_ext = A2 + (A2 - A1) / 3.0
            err = abs(A_ext - A2)

            if err < maxerr:
                return np.float32(A_ext)

            n = n2
            A1 = A2

        return np.float32(A1)

    def _shoelace(self, points):
        """Shoelace formula for polygon area."""
        x, y = points[:, 0], points[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Fit a shape to noisy sample points using Fourier descriptors via FFT.
        """
        start_time = time.time()
        deadline = start_time + maxtime - 0.6

        # Phase 1: Aggressive sampling
        points = []
        while time.time() < deadline - 1.5 and len(points) < 8000:
            try:
                x, y = sample()
                if np.isfinite(x) and np.isfinite(y):
                    points.append([x, y])
            except:
                break

        if len(points) < 12:
            return MyShape()

        points = np.asarray(points, dtype=np.float64)

        # Phase 2: Robust outlier removal
        points = self._remove_outliers(points)
        if len(points) < 12:
            return MyShape()

        # Phase 3: Sort points by angle from centroid
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_points = points[np.argsort(angles)]

        # Phase 4: Resample to uniform arc-length parameterization
        # Power-of-2 for efficient FFT
        if len(sorted_points) < 400:
            N = 512
        elif len(sorted_points) > 4000:
            N = 2048
        else:
            N = 1024

        uniform_pts = self._resample_uniform_arc(sorted_points, N)

        # Phase 5: Fourier descriptors via FFT (fast and stable)
        n_harmonics = min(40, N // 16)
        n_harmonics = max(n_harmonics, 8)

        coeffs_x, coeffs_y = self._fft_fourier_coeffs(uniform_pts, n_harmonics)

        return MyShape(coeffs_x, coeffs_y, n_harmonics)

    def _remove_outliers(self, points):
        """Remove outliers using IQR method (iterative)."""
        if len(points) < 20:
            return points

        pts = points.copy()

        for _ in range(4):
            if len(pts) < 15:
                break

            center = np.median(pts, axis=0)
            dists = np.sqrt(np.sum((pts - center) ** 2, axis=1))

            q1, q3 = np.percentile(dists, [25, 75])
            iqr = q3 - q1
            threshold = q3 + 1.8 * iqr

            mask = dists <= threshold
            if np.sum(mask) < 15:
                break
            pts = pts[mask]

        return pts

    def _resample_uniform_arc(self, points, N):
        """
        Resample polyline to N points equally spaced by arc length.
        Custom implementation without scipy.interpolate.
        """
        P = points
        n_pts = len(P)

        # Compute cumulative arc length
        diffs = np.diff(P, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))

        cum = np.zeros(n_pts, dtype=np.float64)
        cum[1:] = np.cumsum(seg_lengths)

        # Close the curve
        close_vec = P[0] - P[-1]
        close_len = np.sqrt(np.dot(close_vec, close_vec))
        total_len = cum[-1] + close_len

        if total_len < 1e-12:
            return np.tile(P[0], (N, 1))

        # Extended arrays for closed curve
        cum_ext = np.concatenate([cum, [total_len]])
        P_ext = np.vstack([P, P[:1]])

        # Target arc-length positions
        targets = np.linspace(0, total_len, N, endpoint=False)

        # Linear interpolation
        out = np.empty((N, 2), dtype=np.float64)
        j = 0

        for i, s in enumerate(targets):
            # Find interval containing s
            while j + 1 < len(cum_ext) and cum_ext[j + 1] < s:
                j += 1

            s0, s1 = cum_ext[j], cum_ext[j + 1]

            if s1 - s0 < 1e-15:
                out[i] = P_ext[j]
            else:
                t = (s - s0) / (s1 - s0)
                out[i] = (1.0 - t) * P_ext[j] + t * P_ext[j + 1]

        return out

    def _fft_fourier_coeffs(self, points, n_harmonics):
        """
        Fit Fourier descriptors using FFT (fast and numerically stable).
        Output format: [a0, a1, b1, a2, b2, ...]
        """
        x = points[:, 0]
        y = points[:, 1]
        N = len(points)

        # FFT
        fx = np.fft.rfft(x)
        fy = np.fft.rfft(y)

        # Convert FFT coefficients to Fourier series form
        n_coeffs = 2 * n_harmonics + 1
        coeffs_x = np.zeros(n_coeffs, dtype=np.float64)
        coeffs_y = np.zeros(n_coeffs, dtype=np.float64)

        # DC term (mean)
        coeffs_x[0] = fx[0].real / N
        coeffs_y[0] = fy[0].real / N

        # Harmonics
        H = min(n_harmonics, len(fx) - 1)
        for k in range(1, H + 1):
            cx = fx[k] / N
            cy = fy[k] / N

            # x(t) = a0 + Σ (a_k cos(kt) + b_k sin(kt))
            # FFT: a_k = 2*Re(c_k), b_k = -2*Im(c_k)
            coeffs_x[2 * k - 1] = 2.0 * cx.real   # a_k
            coeffs_x[2 * k] = -2.0 * cx.imag      # b_k

            coeffs_y[2 * k - 1] = 2.0 * cy.real   # c_k
            coeffs_y[2 * k] = -2.0 * cy.imag      # d_k

        return coeffs_x, coeffs_y


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
