"""
LARP: Latent Astrophysical Residual Pipeline
============================================

A physics-informed, systematics-marginalized machine learning architecture
for space-based exoplanetary transmission spectroscopy in the ESA Ariel Space Mission
and the NeurIPS Ariel Data Challenge 2025.

Authors:
    Aditya Deshmukh (additya.22311670@viit.ac.in)
    Spandan Kewte (spandan.22311588@viit.ac.in)
    Harshita Kukreja (harshita.22310999@viit.ac.in)
    Omkar Deshpande (omkar.22311757@viit.ac.in)
"""

__version__ = "1.0.0"
__author__ = "Aditya Deshmukh, Spandan Kewte, Harshita Kukreja, Omkar Deshpande"

from larp.metrics import ariel_score, gaussian_log_likelihood, allan_deviation_beta
from larp.calibration import paired_cds_calibration, adc_electron_restoration
from larp.extraction import fit_trace_polynomial, horne_optimal_extraction, compute_spatial_moments
from larp.geometry import mandel_agol_transit, decoupled_white_light_lock
from larp.marginalization import (
    construct_nuisance_basis,
    orthogonalize_doot_common_mode,
    frisch_waugh_lovell_qr_solve,
    propagate_geometry_jacobian,
)
from larp.retrieval import TruncatedSVDRetrieval, woodbury_empirical_bayes
from larp.conformal import compute_allan_factor, SplitConformalCalibrator
from larp.ensemble import MomentMatchedEnsemble, precision_weighted_pool
from larp.pipeline import LARPPipeline

__all__ = [
    "ariel_score",
    "gaussian_log_likelihood",
    "allan_deviation_beta",
    "paired_cds_calibration",
    "adc_electron_restoration",
    "fit_trace_polynomial",
    "horne_optimal_extraction",
    "compute_spatial_moments",
    "mandel_agol_transit",
    "decoupled_white_light_lock",
    "construct_nuisance_basis",
    "orthogonalize_doot_common_mode",
    "frisch_waugh_lovell_qr_solve",
    "propagate_geometry_jacobian",
    "TruncatedSVDRetrieval",
    "woodbury_empirical_bayes",
    "compute_allan_factor",
    "SplitConformalCalibrator",
    "MomentMatchedEnsemble",
    "precision_weighted_pool",
    "LARPPipeline",
]
