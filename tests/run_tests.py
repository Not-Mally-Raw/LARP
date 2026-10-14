"""
LARP Standalone Test Runner (Standard Library unittest)
======================================================
"""

import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from larp.metrics import (
    gaussian_log_likelihood,
    ariel_score,
    allan_deviation_beta,
    get_default_weights,
    FGS1_WEIGHT,
    TOTAL_CHANNELS,
)
from larp.marginalization import (
    orthogonalize_doot_common_mode,
    frisch_waugh_lovell_qr_solve,
)
from larp.conformal import SplitConformalCalibrator, compute_allan_factor
from larp.geometry import mandel_agol_transit


class TestMetrics(unittest.TestCase):
    def test_weights(self):
        w = get_default_weights()
        self.assertEqual(len(w), TOTAL_CHANNELS)
        self.assertEqual(w[0], FGS1_WEIGHT)
        self.assertTrue(np.all(w[1:] == 1.0))

    def test_asymmetric_penalty(self):
        y_true = np.zeros(283)
        y_pred = np.full(283, 0.05)
        gll_over = gaussian_log_likelihood(y_true, y_pred, np.full(283, 0.001))
        gll_under = gaussian_log_likelihood(y_true, y_pred, np.full(283, 0.50))
        self.assertGreater(gll_under, gll_over)


class TestMarginalization(unittest.TestCase):
    def test_doot_orthogonalization(self):
        t = np.linspace(0, 10, 500)
        K = np.exp(-((t - 5.0) ** 2) / 0.5)
        raw_drift = 0.5 * K + 0.1 * t
        c_ortho = orthogonalize_doot_common_mode(raw_drift, K)
        inner_prod = np.dot(c_ortho, K)
        self.assertLess(abs(inner_prod), 1e-10)

    def test_fwl_qr_solve(self):
        n_frames = 1500
        n_channels = 282
        rng = np.random.default_rng(42)
        t = np.linspace(0, 200, n_frames)
        K_1d = np.clip(1.0 - ((t - 100.0) / 30.0) ** 2, 0.0, 1.0)
        K_templates = np.tile(K_1d[:, None], (1, n_channels))
        true_depth = 0.0050

        X_nuis = np.column_stack([np.ones(n_frames)] + [rng.normal(0, 1, n_frames) for _ in range(14)])
        flux_matrix = 1.0 - (true_depth * K_templates) + (X_nuis @ rng.normal(0, 0.001, (15, n_channels)))

        depths, sigmas, solve_time_ms = frisch_waugh_lovell_qr_solve(
            flux_matrix=flux_matrix,
            transit_templates=K_templates,
            X_nuis=X_nuis,
        )
        self.assertLess(abs(np.mean(depths) - true_depth), 0.0001)
        self.assertLess(solve_time_ms, 100.0)


class TestConformal(unittest.TestCase):
    def test_fgs1_floor(self):
        calibrator = SplitConformalCalibrator(target_coverage=0.95, min_fgs1_floor=0.000950)
        y_pred = np.full(283, 0.0050)
        y_sigma = np.full(283, 0.000060)
        cal_sig = calibrator.calibrate(y_pred, y_sigma)
        self.assertGreaterEqual(cal_sig[0], 0.000950)


if __name__ == "__main__":
    unittest.main()
