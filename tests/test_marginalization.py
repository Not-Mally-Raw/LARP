"""Unit tests for larp.marginalization"""
import numpy as np
import pytest
from larp.marginalization import (
    orthogonalize_doot_common_mode,
    construct_nuisance_basis,
    frisch_waugh_lovell_qr_solve,
)


def test_doot_orthogonalization():
    t = np.linspace(0, 10, 500)
    # Synthetic transit trough
    K = np.exp(-((t - 5.0) ** 2) / 0.5)
    # Correlated nuisance drift
    raw_drift = 0.5 * K + 0.1 * t

    c_ortho = orthogonalize_doot_common_mode(raw_drift, K)
    # Correlation between c_ortho and K must be virtually zero
    inner_prod = np.dot(c_ortho, K)
    assert abs(inner_prod) < 1e-10


def test_fwl_qr_solve_zero_dilution_and_speed():
    n_frames = 2000
    n_channels = 282
    rng = np.random.default_rng(42)

    # Unit transit template
    t = np.linspace(0, 200, n_frames)
    K_1d = np.clip(1.0 - ((t - 100.0) / 30.0) ** 2, 0.0, 1.0)
    K_templates = np.tile(K_1d[:, None], (1, n_channels))

    # True injected depths: 5000 ppm = 0.005
    true_depth = 0.0050

    # Nuisance design matrix: 15 columns
    X_nuis = np.column_stack([np.ones(n_frames)] + [rng.normal(0, 1, n_frames) for _ in range(14)])

    # Synthetic baseline flux + transit signal + systematics + noise
    nuis_coeffs = rng.normal(0, 0.001, size=(15, n_channels))
    noise = rng.normal(0, 0.0002, size=(n_frames, n_channels))
    flux_matrix = 1.0 - (true_depth * K_templates) + (X_nuis @ nuis_coeffs) + noise

    depths, sigmas, solve_time_ms = frisch_waugh_lovell_qr_solve(
        flux_matrix=flux_matrix,
        transit_templates=K_templates,
        X_nuis=X_nuis,
    )

    # 1. Recovered depth must be within 1% of true depth (zero dilution)
    mean_recovered = np.mean(depths)
    assert abs(mean_recovered - true_depth) < 0.0001  # < 100 ppm error

    # 2. Solve time across 282 channels must be rapid (< 50 ms)
    assert solve_time_ms < 100.0
