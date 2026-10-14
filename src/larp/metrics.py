"""
LARP Metrics & Loss Module
==========================

Implements the official NeurIPS Ariel Data Challenge 2025 scoring engine,
asymmetric Gaussian Log-Likelihood (GLL), and red-noise Allan factor metrics.
"""

from __future__ import annotations
import numpy as np

FGS1_WEIGHT = 57.846
AIRS_WEIGHT = 1.000
TOTAL_CHANNELS = 283
TOTAL_WEIGHT = FGS1_WEIGHT + 282 * AIRS_WEIGHT  # 339.846


def get_default_weights() -> np.ndarray:
    """Return default channel weights (57.846 on Ch 0, 1.0 on Ch 1-282)."""
    weights = np.ones(TOTAL_CHANNELS, dtype=np.float64)
    weights[0] = FGS1_WEIGHT
    return weights


def gaussian_log_likelihood(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_sigma: np.ndarray,
    weights: np.ndarray | None = None,
    axis: int = -1,
) -> float | np.ndarray:
    """
    Compute weighted Gaussian Log-Likelihood (GLL).

    L = -0.5 * sum_j w_j * [ ln(2*pi*sigma_j^2) + (y_j - mu_j)^2 / sigma_j^2 ]

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth transit depths (R_p / R_*)^2, shape (..., 283).
    y_pred : np.ndarray
        Predicted mean transit depths, shape (..., 283).
    y_sigma : np.ndarray
        Predicted uncertainties, shape (..., 283), strictly positive.
    weights : np.ndarray, optional
        Per-channel weights. Defaults to Ariel 2025 standard weights.
    axis : int
        Axis along which channels are summed.

    Returns
    -------
    gll : float or np.ndarray
        Weighted GLL value(s).
    """
    if weights is None:
        weights = get_default_weights()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_sigma = np.asarray(y_sigma, dtype=np.float64)

    # Protect against non-positive or underflow sigma
    y_sigma = np.clip(y_sigma, 1e-9, None)

    diff_sq = (y_true - y_pred) ** 2
    var = y_sigma ** 2
    log_term = np.log(2.0 * np.pi * var)
    quad_term = diff_sq / var

    weighted_loss = weights * (log_term + quad_term)
    return -0.5 * np.sum(weighted_loss, axis=axis)


def ariel_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_sigma: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    weights: np.ndarray | None = None,
    sigma_ideal: float | np.ndarray = 1e-5,
) -> float:
    """
    Compute the official Normalized Relative Ariel Score (S).

    S = (L - L_ref) / (L_ideal - L_ref)

    where L_ref is the likelihood of the population reference model (mean, std),
    and L_ideal corresponds to perfect prediction with theoretical sensor noise.

    A score of S = 0.0000 corresponds to the population reference.
    A score of S = 1.0000 corresponds to ideal physical extraction.
    """
    if weights is None:
        weights = get_default_weights()

    l_model = float(np.mean(gaussian_log_likelihood(y_true, y_pred, y_sigma, weights)))
    l_ref = float(np.mean(gaussian_log_likelihood(y_true, ref_mean, ref_std, weights)))

    if np.isscalar(sigma_ideal):
        ideal_sig = np.full_like(y_true, sigma_ideal)
        ideal_sig[..., 0] = 1e-6  # 1 ppm ideal on FGS1
    else:
        ideal_sig = np.asarray(sigma_ideal)

    l_ideal = float(np.mean(gaussian_log_likelihood(y_true, y_true, ideal_sig, weights)))

    denominator = l_ideal - l_ref
    if abs(denominator) < 1e-12:
        return 0.0

    return float((l_model - l_ref) / denominator)


def allan_deviation_beta(
    residuals: np.ndarray,
    bin_sizes: tuple[int, ...] = (1, 5, 10, 20),
) -> tuple[dict[int, float], float]:
    """
    Compute the empirical Allan deviation scaling factor beta(N) to detect red noise.

    For pure Gaussian white noise:
        std(binned_N) = std(unbinned) / sqrt(N)  ==>  beta(N) = 1.00

    For correlated red noise / unmarginalized drift:
        beta(N) > 1.35 indicates severe residual systematics.

    Parameters
    ----------
    residuals : np.ndarray
        1D residual time series (or 2D array of residuals).
    bin_sizes : tuple of int
        Bin factors N over which to compute standard deviation.

    Returns
    -------
    stds : dict[int, float]
        Dictionary mapping bin factor N to empirical standard deviation.
    beta_max : float
        The maximum Allan ratio: beta(N) = std_N * sqrt(N) / std_1.
    """
    res = np.asarray(residuals, dtype=np.float64).ravel()
    n_pts = len(res)
    std_1 = float(np.std(res))
    if std_1 == 0:
        return {b: 0.0 for b in bin_sizes}, 1.0

    stds = {1: std_1}
    beta_max = 1.0

    for b in bin_sizes:
        if b == 1:
            continue
        n_bins = n_pts // b
        if n_bins < 2:
            continue
        trimmed = res[: n_bins * b].reshape(n_bins, b)
        binned = np.mean(trimmed, axis=1)
        s_b = float(np.std(binned))
        stds[b] = s_b
        ratio = (s_b * np.sqrt(b)) / std_1
        if ratio > beta_max:
            beta_max = float(ratio)

    return stds, beta_max
