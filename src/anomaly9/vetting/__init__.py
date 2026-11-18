"""
Anomaly9 Vetting Module
=======================
Astrophysical false-positive vetting and candidate scoring engine.
"""

from anomaly9.vetting.checks import (
    VettingCheckResult,
    odd_even_depth_test,
    duration_period_consistency,
    transit_shape_boxiness,
    momentum_dump_proximity,
    signal_to_pink_noise_ratio,
    secondary_eclipse_check,
    run_all_checks,
)

from anomaly9.vetting.scorer import (
    VettingReport,
    compute_confidence_score,
    determine_disposition,
    vet_candidate,
)

__all__ = [
    "VettingCheckResult",
    "odd_even_depth_test",
    "duration_period_consistency",
    "transit_shape_boxiness",
    "momentum_dump_proximity",
    "signal_to_pink_noise_ratio",
    "secondary_eclipse_check",
    "run_all_checks",
    "VettingReport",
    "compute_confidence_score",
    "determine_disposition",
    "vet_candidate",
]