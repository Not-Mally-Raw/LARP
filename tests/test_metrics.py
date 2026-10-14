"""Unit tests for larp.metrics"""
import numpy as np
import pytest
from larp.metrics import (
    gaussian_log_likelihood,
    ariel_score,
    allan_deviation_beta,
    get_default_weights,
    FGS1_WEIGHT,
    TOTAL_CHANNELS,
)


def test_default_weights():
    weights = get_default_weights()
    assert len(weights) == TOTAL_CHANNELS
    assert weights[0] == FGS1_WEIGHT
    assert np.all(weights[1:] == 1.0)


def test_gaussian_log_likelihood_symmetry():
    y_true = np.zeros(283)
    y_sigma = np.ones(283)

    gll_pos = gaussian_log_likelihood(y_true, np.full(283, 0.1), y_sigma)
    gll_neg = gaussian_log_likelihood(y_true, np.full(283, -0.1), y_sigma)
    assert np.isclose(gll_pos, gll_neg)


def test_asymmetric_uncertainty_penalty():
    y_true = np.zeros(283)
    bias = 0.05
    y_pred = np.full(283, bias)

    # Overconfident: sigma = 0.001 (50-sigma error)
    gll_overconfident = gaussian_log_likelihood(y_true, y_pred, np.full(283, 0.001))

    # Underconfident: sigma = 0.50 (mild overestimation)
    gll_underconfident = gaussian_log_likelihood(y_true, y_pred, np.full(283, 0.50))

    # Overconfidence penalty must be vastly more catastrophic than underconfidence
    assert gll_underconfident > gll_overconfident


def test_allan_deviation_pure_white_noise():
    rng = np.random.default_rng(123)
    white_noise = rng.normal(0.0, 1.0, size=10000)
    stds, beta_max = allan_deviation_beta(white_noise, bin_sizes=(1, 5, 10, 20))

    # For pure white noise, beta_max should be close to 1.0 (well below 1.35)
    assert beta_max < 1.20
