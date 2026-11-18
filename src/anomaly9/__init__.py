"""
Anomaly9: Core NASA MAST Data Ingestion, Detrending & Dual Detection Engine
Phase 2: + Astrophysical False-Positive Vetting, Multi-Target Batch Pipeline
Phase 3: + Research-Grade Vetting & Phase-Folded Diagnostics
"""

__version__ = "0.3.0"

from anomaly9.vetting import (
    VettingReport,
    VettingCheckResult,
    vet_candidate,
    run_all_checks,
    odd_even_depth_test,
    duration_period_consistency,
    transit_shape_boxiness,
    momentum_dump_proximity,
    signal_to_pink_noise_ratio,
    secondary_eclipse_check,
    compute_confidence_score,
    determine_disposition,
)

__all__ = [
    "VettingReport",
    "VettingCheckResult",
    "vet_candidate",
    "run_all_checks",
    "odd_even_depth_test",
    "duration_period_consistency",
    "transit_shape_boxiness",
    "momentum_dump_proximity",
    "signal_to_pink_noise_ratio",
    "secondary_eclipse_check",
    "compute_confidence_score",
    "determine_disposition",
]