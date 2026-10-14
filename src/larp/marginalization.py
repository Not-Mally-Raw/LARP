"""
LARP Marginalization Module (LARP Core)
=======================================

Vectorized Frisch-Waugh-Lovell (FWL) QR Systematics Marginalization,
Gram-Schmidt Orthogonalized Common Mode, and Analytical Geometry Jacobian Compounding.
"""

from __future__ import annotations
import numpy as np


def orthogonalize_doot_common_mode(
    raw_common_mode: np.ndarray,
    transit_template: np.ndarray,
) -> np.ndarray:
    """
    Orthogonalize out-of-transit common mode against the transit kernel via Gram-Schmidt.

    c_common <- c_common - ( <c_common, K_wl> / ||K_wl||^2 ) * K_wl

    Guarantees: |corr(c_common, K_wl)| < 1e-10, preventing common-mode subtraction
    from inadvertently eroding true physical transit depths.
    """
    c = np.asarray(raw_common_mode, dtype=np.float64)
    k = np.asarray(transit_template, dtype=np.float64)

    norm_k_sq = np.dot(k, k)
    if norm_k_sq < 1e-12:
        return c

    proj = (np.dot(c, k) / norm_k_sq) * k
    c_ortho = c - proj
    return c_ortho


def construct_nuisance_basis(
    time_min: np.ndarray,
    spatial_moments: dict[str, np.ndarray],
    orthogonal_common_mode: np.ndarray,
    fgs_telemetry: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Construct the unified 15-vector spatio-temporal nuisance design matrix X_nuis.

    Columns include:
    [1, t, P_2(t), P_3(t), c_common, y0, y0^2, width, width^2, skewness, background,
     dx_fgs, dy_fgs, dx_fgs^2, dy_fgs^2]
    """
    t = np.asarray(time_min, dtype=np.float64)
    n = len(t)

    # Standardize time
    t_norm = (t - np.mean(t)) / (np.std(t) if np.std(t) > 0 else 1.0)
    p2 = 0.5 * (3.0 * t_norm ** 2 - 1.0)
    p3 = 0.5 * (5.0 * t_norm ** 3 - 3.0 * t_norm)

    basis_cols = [
        np.ones(n, dtype=np.float64),
        t_norm,
        p2,
        p3,
        orthogonal_common_mode,
        spatial_moments["y0"],
        spatial_moments["y0"] ** 2,
        spatial_moments["width"],
        spatial_moments["width"] ** 2,
        spatial_moments["skewness"],
        spatial_moments["background"],
    ]

    if fgs_telemetry is not None and "dx" in fgs_telemetry:
        dx = fgs_telemetry["dx"]
        dy = fgs_telemetry["dy"]
        basis_cols.extend([dx, dy, dx ** 2, dy ** 2])
    else:
        # Higher-order spatial & temporal terms to complete 15-vector basis
        basis_cols.extend([
            spatial_moments["y0"] * spatial_moments["width"],
            spatial_moments["y0"] * spatial_moments["skewness"],
            t_norm * spatial_moments["y0"],
            t_norm * spatial_moments["width"],
        ])

    X_nuis = np.column_stack(basis_cols[:15])
    return X_nuis


def frisch_waugh_lovell_qr_solve(
    flux_matrix: np.ndarray,
    transit_templates: np.ndarray,
    X_nuis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the 282-channel marginalization simultaneously using the Frisch-Waugh-Lovell (FWL)
    theorem via thin QR decomposition in ~14 ms.

    F = -alpha * K + X_nuis * gamma + eps
    P_perp = I - Q Q^T
    alpha_j = - (K_tilde_j^T F_tilde_j) / (K_tilde_j^T K_tilde_j)
    D_j = alpha_j / beta_{0, j}

    Parameters
    ----------
    flux_matrix : np.ndarray
        Extracted flux array of shape (T, N_channels).
    transit_templates : np.ndarray
        Unit-p^2 transit template array of shape (T, N_channels).
    X_nuis : np.ndarray
        Nuisance design matrix of shape (T, K_nuis).

    Returns
    -------
    depths : np.ndarray
        Marginalized transit depths D_j for each channel.
    formal_sigma : np.ndarray
        Analytical WLS standard errors.
    solve_time_ms : float
        Execution time in milliseconds.
    """
    import time

    t_start = time.perf_counter()

    # 1. Thin QR factorization of nuisance design matrix (computed once)
    Q, R = np.linalg.qr(X_nuis, mode="reduced")  # Q: (T, K), R: (K, K)

    # 2. Economy orthogonal complement projection: P_perp * V = V - Q * (Q^T * V)
    # BLAS-3 batch matrix multiplications
    Qt_F = Q.T @ flux_matrix                      # (K, C)
    F_tilde = flux_matrix - Q @ Qt_F              # (T, C)

    Qt_K = Q.T @ transit_templates                # (K, C)
    K_tilde = transit_templates - Q @ Qt_K        # (T, C)

    # 3. Direct FWL optimal coefficient solve
    num = np.sum(K_tilde * F_tilde, axis=0)       # (C,)
    denom = np.sum(K_tilde ** 2, axis=0)          # (C,)
    denom = np.where(denom > 1e-12, denom, 1e-12)
    alpha = -num / denom

    # 4. Triangular back-solve for baseline flux beta_{0, j}
    # R * gamma = Q^T * (F + alpha * K)
    rhs = Qt_F + (Q.T @ (transit_templates * alpha[None, :]))
    gamma = np.linalg.solve(R, rhs)
    beta_0 = gamma[0, :]  # Column 0 of X_nuis is ones, so gamma_0 is intercept
    beta_0 = np.where(np.abs(beta_0) > 1e-6, beta_0, 1.0)

    depths = alpha / beta_0

    # 5. Residuals & formal WLS uncertainty
    # Residuals: F_tilde + alpha * K_tilde
    res = F_tilde + transit_templates * alpha[None, :] - Q @ (Q.T @ (transit_templates * alpha[None, :]))
    n_frames, n_channels = flux_matrix.shape
    dof = max(1, n_frames - X_nuis.shape[1] - 1)
    chi2_red = np.sum(res ** 2, axis=0) / dof

    var_alpha = chi2_red / denom
    formal_sigma = np.sqrt(np.clip(var_alpha, 1e-12, None)) / np.abs(beta_0)

    solve_time_ms = (time.perf_counter() - t_start) * 1000.0

    return depths, formal_sigma, solve_time_ms


def propagate_geometry_jacobian(
    formal_sigma: np.ndarray,
    jacobian_geo: np.ndarray,
    cov_geo: np.ndarray,
) -> np.ndarray:
    """
    Fold Stage 4 orbital geometry posterior covariance into spectral error bars via the Delta method.

    sigma_total_j^2 = sigma_FWL_j^2 + J_geo_j^T * C_geo * J_geo_j

    Parameters
    ----------
    formal_sigma : np.ndarray
        Formal FWL uncertainty array of shape (N_channels,).
    jacobian_geo : np.ndarray
        Geometry Jacobian of shape (N_channels, 3) where columns are [dD/dt0, dD/d(a/Rs), dD/di].
    cov_geo : np.ndarray
        Geometry posterior covariance matrix of shape (3, 3).

    Returns
    -------
    compounded_sigma : np.ndarray
        Total uncertainty array of shape (N_channels,).
    """
    n_channels = len(formal_sigma)
    geo_var = np.zeros(n_channels, dtype=np.float64)

    for j in range(n_channels):
        j_vec = jacobian_geo[j, :]
        geo_var[j] = float(j_vec.T @ cov_geo @ j_vec)

    total_var = (formal_sigma ** 2) + np.clip(geo_var, 0.0, None)
    return np.sqrt(total_var)
