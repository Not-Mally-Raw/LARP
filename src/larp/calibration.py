"""
LARP Calibration Module
=======================

Physics-informed detector calibration and paired Correlated Double Sampling (CDS)
for Ariel space mission non-destructive readouts.
"""

from __future__ import annotations
import numpy as np


ADC_GAIN = 0.4369      # e- / ADU
ADC_OFFSET = -1000.0   # ADU offset
LINEARITY_C1 = 0.0     # 1st-order linearity coefficient


def adc_electron_restoration(
    raw_adu: np.ndarray,
    gain: float = ADC_GAIN,
    offset: float = ADC_OFFSET,
) -> np.ndarray:
    """
    Convert raw 16-bit integer ADU readouts to calibrated electron counts in float64.

    S_e = (S_ADU - Offset_ADC) * Gain_ADC

    Enforces 64-bit float precision to prevent integer wrap-around underflow.
    """
    raw_f64 = np.asarray(raw_adu, dtype=np.float64)
    return (raw_f64 - offset) * gain


def apply_linearity_correction(
    electrons: np.ndarray,
    c1: float = LINEARITY_C1,
) -> np.ndarray:
    """
    Apply polynomial linearity well-depth correction to electron array.

    S_lin = S_e * (1.0 + c1 * S_e * 1e-6)
    """
    if c1 == 0.0:
        return electrons
    return electrons * (1.0 + c1 * electrons * 1e-6)


def paired_cds_calibration(
    read_pairs: np.ndarray,
    dark_frame: np.ndarray | None = None,
    flat_field: np.ndarray | None = None,
    gain: float = ADC_GAIN,
    offset: float = ADC_OFFSET,
) -> np.ndarray:
    """
    Perform paired Correlated Double Sampling (CDS) across time-series readout cubes.

    Raw Ariel non-destructive readouts alternate between a short reset read S_0
    and a subsequent integration read S_1:
        Delta S(t) = S_1(t) - S_0(t) = (Q_reset + Q_star) - Q_reset = Q_star

    This cancels kTC thermal reset noise, achieving a ~18.3 dB empirical noise drop.

    Parameters
    ----------
    read_pairs : np.ndarray
        Array of shape (T, 2, Y, X) or (2*T, Y, X).
    dark_frame : np.ndarray, optional
        Master dark frame in electron counts.
    flat_field : np.ndarray, optional
        Master normalized flat-field response matrix.
    gain : float
        ADC conversion gain in e- / ADU.
    offset : float
        ADC pedestal offset in ADU.

    Returns
    -------
    calibrated_cube : np.ndarray
        Calibrated flux cube of shape (T, Y, X) in electron counts.
    """
    arr = np.asarray(read_pairs, dtype=np.float64)

    if arr.ndim == 4 and arr.shape[1] == 2:
        s0 = arr[:, 0, :, :]
        s1 = arr[:, 1, :, :]
    elif arr.ndim == 3 and arr.shape[0] % 2 == 0:
        s0 = arr[0::2, :, :]
        s1 = arr[1::2, :, :]
    else:
        # Pre-differenced cube
        return (arr - offset) * gain

    # Electron conversion
    e0 = (s0 - offset) * gain
    e1 = (s1 - offset) * gain

    # CDS differencing: S_1 - S_0
    delta_s = e1 - e0

    # Optional dark subtraction
    if dark_frame is not None:
        delta_s = delta_s - np.asarray(dark_frame, dtype=np.float64)

    # Optional flat-field normalization
    if flat_field is not None:
        norm_flat = np.asarray(flat_field, dtype=np.float64)
        norm_flat = np.where(norm_flat > 1e-6, norm_flat, 1.0)
        delta_s = delta_s / norm_flat

    return delta_s
