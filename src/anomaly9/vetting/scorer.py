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
                    "transit_shape_boxiness": "V_SHAPED_TRANSIT",
                    "momentum_dump_proximity": "MOMENTUM_DUMP_PROXIMITY",
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
    
    Factors (recalibrated for high-SPNR planetary transits):
    - Fraction of passed checks (weighted by importance)
    - BLS SNR (capped at 30, scaled to 0-1)
    - SPNR (capped at 20, scaled to 0-1) - STRONG weight for high SPNR
    - Transit depth significance (typical exoplanets: 100-5000 ppm)
    - Period plausibility (0.5-50 days typical for planets)
    - Duration consistency (duration/period ratio)
    - Bonus for high SPNR + clean critical checks
    """
    if not check_results:
        return 0.0
    
    # --- Identify key checks ---
    check_map = {c.name: c for c in check_results}
    spnr_check = check_map.get("signal_to_pink_noise")
    odd_even_check = check_map.get("odd_even_depth")
    secondary_check = check_map.get("secondary_eclipse")
    duration_check = check_map.get("duration_period_consistency")
    shape_check = check_map.get("transit_shape_boxiness")
    momentum_check = check_map.get("momentum_dump_proximity")
    
    spnr = spnr_check.value if spnr_check else 0.0
    
    # --- Base confidence from passed checks (weighted by importance) ---
    # Critical checks (high weight): odd_even, secondary_eclipse, duration_period
    # Important checks (medium weight): transit_shape_boxiness, signal_to_pink_noise
    # Advisory checks (low weight): momentum_dump_proximity
    weights_checks = {
        "odd_even_depth": 0.25,
        "secondary_eclipse": 0.25,
        "duration_period_consistency": 0.15,
        "transit_shape_boxiness": 0.15,
        "signal_to_pink_noise": 0.10,
        "momentum_dump_proximity": 0.10,
    }
    
    check_score = 0.0
    total_weight = 0.0
    for check in check_results:
        w = weights_checks.get(check.name, 0.10)
        if check.passed:
            check_score += w
        total_weight += w
    check_score = check_score / total_weight if total_weight > 0 else 0.0
    
    # --- SNR contribution (capped at 30, scaled to 0-1) ---
    snr_score = min(snr / 30.0, 1.0) if snr > 0 else 0.0
    
    # --- SPNR contribution (STRONG weight for high SPNR) ---
    # SPNR > 15 is excellent, 10-15 good, 7-10 marginal, <7 poor
    if spnr >= 15:
        spnr_score = 1.0
    elif spnr >= 10:
        spnr_score = 0.8 + 0.2 * (spnr - 10) / 5  # 0.8 to 1.0
    elif spnr >= 7:
        spnr_score = 0.5 + 0.3 * (spnr - 7) / 3  # 0.5 to 0.8
    else:
        spnr_score = 0.5 * spnr / 7  # 0 to 0.5
    
    # --- Depth significance ---
    # Typical exoplanet depths: 100-5000 ppm. Jupiter ~10000 ppm.
    # Score peaks around 1000-5000 ppm
    if depth_ppm <= 0:
        depth_score = 0.0
    elif depth_ppm < 100:
        depth_score = 0.3  # Very shallow, could be noise
    elif depth_ppm < 500:
        depth_score = 0.7
    elif depth_ppm < 5000:
        depth_score = 1.0
    elif depth_ppm < 15000:
        depth_score = 0.8  # Very deep, could be EB
    else:
        depth_score = 0.3  # Too deep for typical planet
    
    # --- Period plausibility ---
    # Hot Jupiters: 0.5-10 days, Warm: 10-100, Earth-like: >100 (but hard to detect)
    if 0.5 <= period <= 50:
        period_score = 1.0
    elif 0.3 <= period < 0.5:
        period_score = 0.8  # Ultra-short period possible
    elif 50 < period <= 100:
        period_score = 0.7
    else:
        period_score = 0.4
    
    # --- Duration consistency ---
    dur_per = duration / period if period > 0 else 0
    if 0.005 <= dur_per <= 0.05:
        duration_score = 1.0
    elif 0.001 <= dur_per < 0.005:
        duration_score = 0.7
    elif 0.05 < dur_per <= 0.1:
        duration_score = 0.7
    else:
        duration_score = 0.4
    
    # --- Weighted combination ---
    # Key: SPNR and check_score are the primary drivers
    weights = {
        "checks": 0.35,
        "spnr": 0.25,
        "snr": 0.15,
        "depth": 0.10,
        "period": 0.08,
        "duration": 0.07,
    }
    
    confidence = (
        weights["checks"] * check_score +
        weights["spnr"] * spnr_score +
        weights["snr"] * snr_score +
        weights["depth"] * depth_score +
        weights["period"] * period_score +
        weights["duration"] * duration_score
    )
    
    # --- Penalties for specific failure modes ---
    # Only penalize for genuine astrophysical false positive indicators
    # NOT for instrumental artifacts (momentum dump proximity is advisory)
    
    critical_failures = 0
    if odd_even_check and not odd_even_check.passed:
        critical_failures += 1
    if secondary_check and not secondary_check.passed:
        critical_failures += 1
    if duration_check and not duration_check.passed:
        critical_failures += 1
    if shape_check and not shape_check.passed:
        critical_failures += 1
    
    # Apply penalty only for astrophysical false positives
    if critical_failures >= 2:
        confidence *= 0.3
    elif critical_failures == 1:
        confidence *= 0.6
    
    # --- Bonus for high-confidence planetary signatures ---
    # High SPNR + clean critical checks = strong planetary candidate
    clean_critical = (odd_even_check and odd_even_check.passed and
                      secondary_check and secondary_check.passed and
                      duration_check and duration_check.passed)
    
    if clean_critical and spnr >= 15 and snr >= 15:
        confidence = min(confidence * 1.15, 1.0)  # Boost for excellent candidates
    elif clean_critical and spnr >= 10 and snr >= 10:
        confidence = min(confidence * 1.1, 1.0)   # Boost for good candidates
    
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
    # Only EB_SUSPECT and V_SHAPED_TRANSIT are hard false positive indicators
    # MOMENTUM_DUMP_PROXIMITY and LOW_SPNR are advisory/warning flags
    hard_fp_flags = {"EB_SUSPECT", "V_SHAPED_TRANSIT"}
    advisory_flags = {"MOMENTUM_DUMP_PROXIMITY", "LOW_SPNR", "DURATION_ANOMALY"}
    
    has_hard_fp = any(f in flags for f in hard_fp_flags)
    has_advisory = any(f in flags for f in advisory_flags)
    
    # Count failed astrophysical critical checks
    astrophysical_failures = sum(1 for c in check_results 
                                if not c.passed and c.name in 
                                {"odd_even_depth", "secondary_eclipse", "duration_period_consistency", "transit_shape_boxiness"})
    
    if has_hard_fp:
        if confidence < 0.3:
            return "FALSE_POSITIVE"
        elif confidence < 0.5:
            return "AMBIGUOUS"
        else:
            # High confidence despite flag - likely borderline
            return "AMBIGUOUS"
    elif astrophysical_failures >= 2:
        return "FALSE_POSITIVE"
    elif confidence >= 0.85 and astrophysical_failures == 0:
        return "CANDIDATE"
    elif confidence >= 0.65:
        return "CANDIDATE"
    elif confidence >= 0.45:
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