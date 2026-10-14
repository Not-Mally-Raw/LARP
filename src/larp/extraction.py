"""
LARP Extraction Module
======================

Curved optical trace determination, Horne (1986) optimal spectral extraction,
and 4-moment spatial PSF telemetry tracking.
"""

from __future__ import annotations
import numpy as np


def fit_trace_polynomial(
    frame: np.ndarray,
    dispersion_cols: tuple[int, int] = (37, 319),
    order: int = 2,
) -> np.ndarray:
    """
    Fit a polynomial trajectory y_trace(x) = sum_k c_k * x^k along the curved spectral trace.

    Parameters
    ----------
    frame : np.ndarray
        2D detector image (Y, X).
    dispersion_cols : tuple of int
        (x_start, x_end) column range containing the optical spectrum.
    order : int
        Polynomial order (typically 2 for Ariel prism dispersion).

    Returns
    -------
    coeffs : np.ndarray
        Polynomial coefficients [c_k, ..., c_0] compatible with np.polyval.
    """
    x_start, x_end = dispersion_cols
    x_indices = np.arange(x_start, x_end)
    y_centroids = []

    y_coords = np.arange(frame.shape[0])

    for x in x_indices:
        column_flux = frame[:, x]
        flux_pos = np.clip(column_flux - np.median(column_flux), 0, None)
        total = np.sum(flux_pos)
        if total > 0:
            y_c = np.sum(y_coords * flux_pos) / total
        else:
            y_c = float(frame.shape[0] / 2.0)
        y_centroids.append(y_c)

    coeffs = np.polyfit(x_indices, y_centroids, deg=order)
    return coeffs


def horne_optimal_extraction(
    cube: np.ndarray,
    coeffs: np.ndarray,
    dispersion_cols: tuple[int, int] = (37, 319),
    half_aperture: int = 6,
    read_noise: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 1D spectra from a 3D calibrated cube using Horne (1986) variance-weighted optimal extraction.

    F(t, x) = sum_y [ P(y, x) * (D - B) / V ] / sum_y [ P^2(y, x) / V ]
    sigma_F^2(t, x) = 1 / sum_y [ P^2(y, x) / V ]

    Parameters
    ----------
    cube : np.ndarray
        Calibrated data cube of shape (T, Y, X).
    coeffs : np.ndarray
        Trace polynomial coefficients.
    dispersion_cols : tuple of int
        (x_start, x_end) column range.
    half_aperture : int
        Aperture radius in pixels around the trace center.
    read_noise : float
        Detector readout noise in electrons.

    Returns
    -------
    flux : np.ndarray
        Extracted flux of shape (T, N_channels).
    variance : np.ndarray
        Formal variance of shape (T, N_channels).
    """
    n_frames, n_y, _ = cube.shape
    x_start, x_end = dispersion_cols
    n_channels = x_end - x_start

    x_grid = np.arange(x_start, x_end)
    y_centers = np.polyval(coeffs, x_grid)

    extracted_flux = np.zeros((n_frames, n_channels), dtype=np.float64)
    extracted_var = np.zeros((n_frames, n_channels), dtype=np.float64)

    # Time-averaged profile for extraction weights
    mean_frame = np.median(cube, axis=0)

    for i, (x, y_c) in enumerate(zip(x_grid, y_centers)):
        y_low = max(0, int(np.floor(y_c - half_aperture)))
        y_high = min(n_y, int(np.ceil(y_c + half_aperture + 1)))

        profile_slice = mean_frame[y_low:y_high, x]
        bg = np.median(mean_frame[:, x])
        p_sub = np.clip(profile_slice - bg, 0, None)
        p_norm = p_sub / np.sum(p_sub) if np.sum(p_sub) > 0 else np.ones_like(p_sub) / len(p_sub)

        # Slice time-series
        data_slice = cube[:, y_low:y_high, x]  # (T, Y_cut)
        d_sub = data_slice - bg

        v = np.clip(data_slice, 0, None) + read_noise ** 2
        p_mat = p_norm[None, :]  # (1, Y_cut)

        num = np.sum((p_mat * d_sub) / v, axis=1)
        denom = np.sum((p_mat ** 2) / v, axis=1)
        denom = np.where(denom > 1e-12, denom, 1e-12)

        extracted_flux[:, i] = num / denom
        extracted_var[:, i] = 1.0 / denom

    return extracted_flux, extracted_var


def compute_spatial_moments(
    cube: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Extract frame-by-frame 4-moment spatial PSF telemetry collapsed across dispersion.

    Phi(t, y) = sum_x D(t, y, x)
    - Centroid: y0(t) = sum_y [ y * Phi ] / sum_y Phi
    - Width: w(t) = sqrt( sum_y [ (y - y0)^2 * Phi ] / sum_y Phi )
    - Skewness: s(t) = sum_y [ (y - y0)^3 * Phi ] / (w^3 * sum_y Phi)
    - Background: B(t) = median of outer spatial boundaries

    Returns
    -------
    telemetry : dict of np.ndarray
        Dictionary with keys 'y0', 'width', 'skewness', 'background'.
    """
    collapsed = np.sum(cube, axis=2)  # shape (T, Y)
    n_frames, n_y = collapsed.shape
    y_coords = np.arange(n_y, dtype=np.float64)

    total_flux = np.sum(collapsed, axis=1, keepdims=True)
    total_flux = np.where(total_flux > 1e-6, total_flux, 1.0)
    norm_phi = collapsed / total_flux

    # 1. Centroid
    y0 = np.sum(y_coords[None, :] * norm_phi, axis=1)

    # 2. Width
    diff = y_coords[None, :] - y0[:, None]
    diff_sq = diff ** 2
    width = np.sqrt(np.clip(np.sum(diff_sq * norm_phi, axis=1), 1e-6, None))

    # 3. Skewness
    diff_cubed = diff ** 3
    skew = np.sum(diff_cubed * norm_phi, axis=1) / (width ** 3)

    # 4. Background
    bg = np.median(collapsed[:, :5], axis=1)

    return {
        "y0": y0,
        "width": width,
        "skewness": skew,
        "background": bg,
    }
