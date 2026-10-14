"""Unit tests for larp.conformal"""
import numpy as np
import pytest
from larp.conformal import SplitConformalCalibrator, compute_allan_factor


def test_conformal_fgs1_floor_protection():
    calibrator = SplitConformalCalibrator(target_coverage=0.95, min_fgs1_floor=0.000950)

    y_pred = np.full(283, 0.0050)
    # Give an overconfident FGS1 sigma of 60 ppm (0.000060)
    y_sigma = np.full(283, 0.000060)

    calibrated_sigma = calibrator.calibrate(y_pred, y_sigma, j_magnitude=11.0)

    # FGS1 channel 0 MUST be floored to 950 ppm
    assert calibrated_sigma[0] >= 0.000950
    # Infrared channels must be scaled by stratum multiplier
    assert calibrated_sigma[1] > y_sigma[1]


def test_allan_factor_calculation():
    res = np.random.normal(0, 1, 5000)
    beta = compute_allan_factor(res, bin_size=20)
    assert 0.8 <= beta <= 1.35
