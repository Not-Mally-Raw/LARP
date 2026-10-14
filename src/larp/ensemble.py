"""
LARP Ensemble Module
====================

Multi-Family Moment-Matched Ensembling and Bootstrap Paired Likelihood-Ratio Gate.
"""

from __future__ import annotations
import numpy as np
from larp.metrics import gaussian_log_likelihood


def precision_weighted_pool(
    predictions: list[np.ndarray],
    uncertainties: list[np.ndarray],
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse predictions from diverse model families via precision-weighted pooling and moment matching.

    Lambda_m = 1 / sigma_m^2
    mu_ens = sum_m (w_m * Lambda_m * mu_m) / sum_m (w_m * Lambda_m)
    sigma_aleatoric^2 = 1 / sum_m (w_m * Lambda_m)
    sigma_epistemic^2 = sum_m w_norm_m * (mu_m - mu_ens)^2
    sigma_ens^2 = sigma_aleatoric^2 + sigma_epistemic^2

    Returns
    -------
    mu_ens : np.ndarray
        Blended mean spectrum.
    sigma_ens : np.ndarray
        Moment-matched total uncertainty.
    """
    n_models = len(predictions)
    if weights is None:
        model_weights = np.ones(n_models, dtype=np.float64) / n_models
    else:
        model_weights = np.array(weights, dtype=np.float64)
        model_weights /= np.sum(model_weights)

    preds = [np.asarray(p, dtype=np.float64) for p in predictions]
    sigs = [np.clip(np.asarray(s, dtype=np.float64), 1e-9, None) for s in uncertainties]

    # Pre-allocate
    precisions = [w / (s ** 2) for w, s in zip(model_weights, sigs)]
    sum_prec = np.sum(precisions, axis=0)

    # Weighted mean
    weighted_preds = [prec * p for prec, p in zip(precisions, preds)]
    mu_ens = np.sum(weighted_preds, axis=0) / sum_prec

    # Aleatoric uncertainty
    aleatoric_var = 1.0 / sum_prec

    # Epistemic model-spread variance
    epistemic_var = np.zeros_like(mu_ens)
    for w, p in zip(model_weights, preds):
        epistemic_var += w * ((p - mu_ens) ** 2)

    sigma_ens = np.sqrt(aleatoric_var + epistemic_var)
    return mu_ens, sigma_ens


class MomentMatchedEnsemble:
    """
    Evaluates multi-family candidates (Physics-Informed, RidgeCV, Truncated SVD)
    and verifies Gate T12 (paired bootstrap delta-GLL > 0) before exporting.
    """

    def __init__(self, n_bootstrap: int = 1000, alpha: float = 0.05):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

    def evaluate_gate_t12(
        self,
        y_true: np.ndarray,
        single_pred: np.ndarray,
        single_sig: np.ndarray,
        ens_pred: np.ndarray,
        ens_sig: np.ndarray,
    ) -> tuple[bool, float, tuple[float, float]]:
        """
        Paired Bootstrap Likelihood Ratio Test (Gate T12).

        Returns
        -------
        passed : bool
            True if 95% CI lower bound of delta-GLL (Ensemble - Single) > 0.
        delta_mean : float
            Mean likelihood gain in nats.
        ci : tuple of float
            (ci_low, ci_high) bounds.
        """
        n_samples = len(y_true)
        gll_single = gaussian_log_likelihood(y_true, single_pred, single_sig)
        gll_ens = gaussian_log_likelihood(y_true, ens_pred, ens_sig)

        diff = gll_ens - gll_single
        delta_mean = float(np.mean(diff))

        # Bootstrap confidence interval
        rng = np.random.default_rng(42)
        boot_diffs = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n_samples, size=n_samples)
            boot_diffs.append(float(np.mean(diff[idx])))

        ci_low = float(np.percentile(boot_diffs, 100.0 * (self.alpha / 2.0)))
        ci_high = float(np.percentile(boot_diffs, 100.0 * (1.0 - self.alpha / 2.0)))

        passed = ci_low > 0.0
        return passed, delta_mean, (ci_low, ci_high)
