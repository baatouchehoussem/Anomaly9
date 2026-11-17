"""
Anomaly9 Vetting Scorer
=======================
Aggregates vetting check results into a structured VettingReport
with disposition and confidence scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

import numpy as np

from anomaly9.vetting.checks import VettingCheckResult, run_all_checks

logger = logging.getLogger("anomaly9.vetting.scorer")


@dataclass
class VettingReport:
    """
    Structured vetting report for a candidate signal.
    
    Attributes
    ----------
    tic_id : str
        TESS Input Catalog identifier
    sector : str
        TESS sector number
    disposition : str
        Final disposition: 'CANDIDATE', 'FALSE_POSITIVE', or 'AMBIGUOUS'
    confidence_score : float
        Overall confidence 0.0-1.0 (higher = more likely planetary)
    vetting_flags : List[str]
        List of triggered flag names (e.g., 'EB_SUSPECT', 'INSTRUMENTAL_ARTIFACT')
    metrics : Dict[str, float]
        Dictionary of all check metrics and values
    check_details : List[Dict]
        Detailed results from each vetting check
    """
    tic_id: str
    sector: str
    disposition: str
    confidence_score: float
    vetting_flags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    check_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert numpy types to Python native types for JSON serialization
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        return convert(d)

    @classmethod
    def from_checks(
        cls,
        tic_id: str,
        sector: str,
        period: float,
        t0: float,
        duration: float,
        depth_ppm: float,
        snr: float,
        check_results: List[VettingCheckResult],
    ) -> "VettingReport":
        """
        Create a VettingReport from vetting check results.
        
        Parameters
        ----------
        tic_id : str
            TIC identifier
        sector : str
            Sector number
        period : float
            Orbital period [days]
        t0 : float
            Mid-transit time [BJD]
        duration : float
            Transit duration [days]
        depth_ppm : float
            Transit depth [ppm]
        snr : float
            BLS signal-to-noise ratio
        check_results : List[VettingCheckResult]
            Results from all vetting checks
            
        Returns
        -------
        VettingReport
            Aggregated vetting report with disposition and confidence
        """
        # Collect flags from failed checks
        flags = []
        metrics = {}
        check_details = []
        
        for check in check_results:
            check_dict = {
                "name": check.name,
                "passed": check.passed,
                "value": check.value,
                "threshold": check.threshold,
                "description": check.description,
            }
            check_details.append(check_dict)
            metrics[check.name] = check.value
            
            if not check.passed:
                # Map check names to standard flag names
                flag_map = {
                    "odd_even_depth": "EB_SUSPECT",
                    "duration_period_consistency": "DURATION_ANOMALY",
                    "box_shape_consistency": "V_SHAPED_TRANSIT",
                    "momentum_dump_proximity": "INSTRUMENTAL_ARTIFACT",
                    "signal_to_pink_noise": "LOW_SPNR",
                    "secondary_eclipse": "EB_SUSPECT",
                }
                flag = flag_map.get(check.name, check.name.upper())
                if flag not in flags:
                    flags.append(flag)
        
        # Compute confidence score
        confidence = compute_confidence_score(check_results, snr, depth_ppm, period, duration)
        
        # Determine disposition
        disposition = determine_disposition(flags, confidence, check_results)
        
        # Add core metrics
        metrics.update({
            "period_d": period,
            "t0_bjd": t0,
            "duration_d": duration,
            "depth_ppm": depth_ppm,
            "snr": snr,
        })
        
        return cls(
            tic_id=tic_id,
            sector=sector,
            disposition=disposition,
            confidence_score=confidence,
            vetting_flags=flags,
            metrics=metrics,
            check_details=check_details,
        )


def compute_confidence_score(
    check_results: List[VettingCheckResult],
    snr: float,
    depth_ppm: float,
    period: float,
    duration: float,
) -> float:
    """
    Compute overall confidence score (0.0-1.0).
    
    Factors:
    - Fraction of passed checks (weighted)
    - BLS SNR (capped)
    - SPNR if available
    - Transit depth significance
    """
    if not check_results:
        return 0.0
    
    # Base score from passed checks
    n_passed = sum(1 for c in check_results if c.passed)
    n_total = len(check_results)
    check_score = n_passed / n_total if n_total > 0 else 0.0
    
    # SNR contribution (capped at 20, scaled to 0-1)
    snr_score = min(snr / 20.0, 1.0) if snr > 0 else 0.0
    
    # SPNR contribution if available
    spnr_score = 0.0
    for c in check_results:
        if c.name == "signal_to_pink_noise":
            spnr_score = min(c.value / 15.0, 1.0) if c.value > 0 else 0.0
            break
    
    # Depth significance (deeper transits more significant, but cap at 10000 ppm)
    depth_score = min(depth_ppm / 10000.0, 1.0) if depth_ppm > 0 else 0.0
    
    # Period plausibility (0.5-50 days typical for planets)
    period_score = 1.0
    if period < 0.3 or period > 100:
        period_score = 0.3
    elif period < 0.5 or period > 50:
        period_score = 0.7
    
    # Duration consistency (duration/period ratio)
    dur_per = duration / period if period > 0 else 0
    duration_score = 1.0
    if dur_per < 0.001 or dur_per > 0.1:
        duration_score = 0.5
    
    # Weighted combination
    weights = {
        "checks": 0.40,
        "snr": 0.20,
        "spnr": 0.15,
        "depth": 0.10,
        "period": 0.10,
        "duration": 0.05,
    }
    
    confidence = (
        weights["checks"] * check_score +
        weights["snr"] * snr_score +
        weights["spnr"] * spnr_score +
        weights["depth"] * depth_score +
        weights["period"] * period_score +
        weights["duration"] * duration_score
    )
    
    # Penalize for critical flags
    critical_flags = {"EB_SUSPECT", "INSTRUMENTAL_ARTIFACT", "V_SHAPED_TRANSIT"}
    triggered_critical = sum(1 for c in check_results if not c.passed and 
                            c.name in {"odd_even_depth", "secondary_eclipse", "momentum_dump_proximity", "box_shape_consistency"})
    
    if triggered_critical >= 2:
        confidence *= 0.3
    elif triggered_critical == 1:
        confidence *= 0.6
    
    return float(np.clip(confidence, 0.0, 1.0))


def determine_disposition(
    flags: List[str],
    confidence: float,
    check_results: List[VettingCheckResult],
) -> str:
    """
    Determine final disposition based on flags and confidence.
    
    Returns
    -------
    str
        'CANDIDATE', 'FALSE_POSITIVE', or 'AMBIGUOUS'
    """
    # Hard false positive flags
    fp_flags = {"EB_SUSPECT", "INSTRUMENTAL_ARTIFACT", "V_SHAPED_TRANSIT"}
    has_fp = any(f in flags for f in fp_flags)
    
    # Count failed critical checks
    critical_failures = sum(1 for c in check_results 
                           if not c.passed and c.name in 
                           {"odd_even_depth", "secondary_eclipse", "momentum_dump_proximity", "box_shape_consistency"})
    
    if has_fp and confidence < 0.4:
        return "FALSE_POSITIVE"
    elif has_fp and confidence < 0.6:
        return "AMBIGUOUS"
    elif confidence >= 0.7 and critical_failures == 0:
        return "CANDIDATE"
    elif confidence >= 0.5:
        return "AMBIGUOUS"
    else:
        return "FALSE_POSITIVE"


def vet_candidate(
    tic_id: str,
    sector: str,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    depth_ppm: float,
    snr: float,
    stellar_radius_rsun: Optional[float] = None,
) -> VettingReport:
    """
    Complete vetting pipeline for a single candidate.
    
    Parameters
    ----------
    tic_id : str
        TIC identifier
    sector : str
        TESS sector
    time : np.ndarray
        Time array [BJD]
    flux : np.ndarray
        Detrended flux array
    period : float
        Orbital period [days]
    t0 : float
        Mid-transit time [BJD]
    duration : float
        Transit duration [days]
    depth_ppm : float
        Transit depth [ppm]
    snr : float
        BLS SNR
    stellar_radius_rsun : float, optional
        Stellar radius for physical consistency
        
    Returns
    -------
    VettingReport
        Complete vetting report
    """
    logger.info("Running vetting checks for %s (sector %s)", tic_id, sector)
    
    check_results = run_all_checks(
        time=time,
        flux=flux,
        period=period,
        t0=t0,
        duration=duration,
        depth_ppm=depth_ppm,
        sector=int(sector) if sector.isdigit() else None,
        stellar_radius_rsun=stellar_radius_rsun,
    )
    
    report = VettingReport.from_checks(
        tic_id=tic_id,
        sector=sector,
        period=period,
        t0=t0,
        duration=duration,
        depth_ppm=depth_ppm,
        snr=snr,
        check_results=check_results,
    )
    
    logger.info(
        "Vetting complete for %s: disposition=%s, confidence=%.3f, flags=%s",
        tic_id, report.disposition, report.confidence_score, report.vetting_flags
    )
    
    return report