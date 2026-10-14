"""
LARP Retrieval Module
=====================

Truncated SVD Population Prior and Fast Woodbury Matrix Identity Empirical Bayes Shrinkage.
"""

from __future__ import annotations
import numpy as np


class TruncatedSVDRetrieval:
    """
    Empirical Bayes atmospheric retrieval using a truncated SVD spectral basis
    fitted across the training planetary population.
    """

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.mean_spectrum: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    def fit(self, population_spectra: np.ndarray) -> TruncatedSVDRetrieval:
        """
        Fit population spectral basis from training spectra matrix of shape (N_planets, 283).
        """
        X = np.asarray(population_spectra, dtype=np.float64)
        self.mean_spectrum = np.mean(X, axis=0)
        X_centered = X - self.mean_spectrum

        # Economy SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[: self.n_components, :]       # (K, 283)
        self.singular_values_ = S[: self.n_components]
        return self

    def project(self, spectrum: np.ndarray) -> np.ndarray:
        """Project centered spectrum onto leading SVD modes."""
        if self.components_ is None or self.mean_spectrum is None:
            raise ValueError("Model must be fitted before projection.")
        centered = spectrum - self.mean_spectrum
        return self.components_ @ centered

    def reconstruct(self, weights: np.ndarray) -> np.ndarray:
        """Reconstruct spectrum from latent weights."""
        if self.components_ is None or self.mean_spectrum is None:
            raise ValueError("Model must be fitted before reconstruction.")
        return self.mean_spectrum + (weights @ self.components_)


def woodbury_empirical_bayes(
    y_obs: np.ndarray,
    sigma_obs: np.ndarray,
    basis_modes: np.ndarray,
    mode_prior_var: np.ndarray,
    prior_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rapid Bayesian regularized posterior mean and covariance via Woodbury Matrix Identity.

    Instead of inverting a full (283 x 283) matrix, Woodbury projects onto the
    (K x K) latent subspace (where K = 8):
        (V Lambda V^T + Sigma_n)^(-1) = Sigma_n^(-1) - Sigma_n^(-1) V (Lambda^(-1) + V^T Sigma_n^(-1) V)^(-1) V^T Sigma_n^(-1)

    Parameters
    ----------
    y_obs : np.ndarray
        Noisy observed spectrum, shape (283,).
    sigma_obs : np.ndarray
        Observed noise standard deviation, shape (283,).
    basis_modes : np.ndarray
        Empirical SVD modes V of shape (K, 283).
    mode_prior_var : np.ndarray
        Prior variance for each mode Lambda, shape (K,).
    prior_mean : np.ndarray
        Prior mean spectrum, shape (283,).

    Returns
    -------
    y_post : np.ndarray
        Shrunk posterior mean spectrum, shape (283,).
    sigma_post : np.ndarray
        Posterior uncertainty, shape (283,).
    """
    K, C = basis_modes.shape
    inv_sigma_sq = 1.0 / np.clip(sigma_obs ** 2, 1e-12, None)  # (C,)

    # V^T in math corresponds to basis_modes.T of shape (C, K)
    V = basis_modes.T  # (C, K)

    # 1. Lambda^(-1) + V^T * Sigma_n^(-1) * V  -->  shape (K, K)
    inv_lambda = np.diag(1.0 / np.clip(mode_prior_var, 1e-12, None))
    core_mat = inv_lambda + (V.T * inv_sigma_sq[None, :]) @ V  # (K, K)

    # 2. Invert the small (8 x 8) core matrix
    inv_core = np.linalg.inv(core_mat)  # (K, K)

    # 3. Posterior mean:
    delta_y = y_obs - prior_mean
    # term1 = Sigma_n^(-1) * delta_y
    t1 = inv_sigma_sq * delta_y
    # term2 = Sigma_n^(-1) * V * inv_core * V^T * t1
    v_t1 = V.T @ t1                     # (K,)
    core_v_t1 = inv_core @ v_t1         # (K,)
    correction = (V @ core_v_t1) * inv_sigma_sq

    # Effective precision projection
    w_weights = V @ (inv_core @ (V.T * inv_sigma_sq[None, :]))
    shrunk_delta = delta_y - (V @ core_v_t1)

    y_post = prior_mean + (V @ (inv_core @ (V.T @ (inv_sigma_sq * delta_y))))

    # 4. Posterior variance diagonal
    # diag( (Sigma_n^(-1) + V Lambda^(-1) V^T)^(-1) )
    post_var = (sigma_obs ** 2) - np.sum((V @ inv_core) * V, axis=1) * (sigma_obs ** 2)
    sigma_post = np.sqrt(np.clip(post_var, 1e-12, None))

    return y_post, sigma_post
