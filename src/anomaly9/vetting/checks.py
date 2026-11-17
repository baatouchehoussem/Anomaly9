"""
Anomaly9 Vetting Checks
=======================
Modular false-positive vetting checks for TESS light curve anomalies.

Implements standard astrophysical false-positive diagnostics:
- Odd/Even Transit Depth Test (eclipsing binary flag)
- Duration-to-Period & Box-Shape Consistency (grazing eclipse vs planetary)
- Instrumental / Momentum Dump Proximity Filter
- Signal-to-Pink-Noise Ratio (SPNR) for correlated red noise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("anomaly9.vetting.checks")


@dataclass
class VettingCheckResult:
    """Result of a single vetting check."""
    name: str
    passed: bool
    value: float
    threshold: float
    description: str


# TESS momentum dump cadences (approximate, sector-dependent)
# These are typical cadence indices where momentum dumps occur
# In practice, these would come from TESS data release notes
MOMENTUM_DUMP_CADENCES = {
    # sector: list of cadence indices (0-based) near momentum dumps
    # These are approximate; real values vary by sector
}


def odd_even_depth_test(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    min_transits: int = 3,
) -> VettingCheckResult:
    """
    Odd/Even Transit Depth Test for eclipsing binary detection.

    Compares depths of odd-numbered vs even-numbered transits.
    Significant difference suggests eclipsing binary (primary/secondary eclipses).
    
    Parameters
    ----------
    time : np.ndarray
        Time array [BJD]
    flux : np.ndarray
        Detrended flux array (normalized ~1.0)
    period : float
        Orbital period [days]
    t0 : float
        Reference mid-transit time [BJD]
    duration : float
        Transit duration [days]
    min_transits : int
        Minimum number of transits required for test

    Returns
    -------
    VettingCheckResult
        passed=True if depths are consistent (planetary), 
        passed=False if odd/even differ significantly (EB suspect)
    """
    if len(time) < 10:
        return VettingCheckResult(
            name="odd_even_depth",
            passed=True,
            value=0.0,
            threshold=0.3,
            description="Insufficient data for odd/even test",
        )

    # Phase fold and identify individual transits
    phase = ((time - t0) % period) / period
    in_transit = phase < (duration / period)
    
    # Find transit centers by grouping contiguous in-transit points
    transit_indices = []
    current_group = []
    for i, flag in enumerate(in_transit):
        if flag:
            current_group.append(i)
        elif current_group:
            transit_indices.append(current_group)
            current_group = []
    if current_group:
        transit_indices.append(current_group)

    if len(transit_indices) < min_transits:
        return VettingCheckResult(
            name="odd_even_depth",
            passed=True,
            value=0.0,
            threshold=0.3,
            description=f"Only {len(transit_indices)} transits found (need {min_transits})",
        )

    # Compute depth for each transit
    depths = []
    for group in transit_indices:
        if len(group) < 2:
            continue
        in_flux = flux[group]
        out_mask = ~in_transit
        if out_mask.sum() < 10:
            continue
        out_flux = flux[out_mask]
        depth = np.median(out_flux) - np.median(in_flux)
        depths.append(depth)

    if len(depths) < min_transits:
        return VettingCheckResult(
            name="odd_even_depth",
            passed=True,
            value=0.0,
            threshold=0.3,
            description="Insufficient valid transits for depth measurement",
        )

    depths = np.array(depths)
    odd_depths = depths[0::2]   # 1st, 3rd, 5th... (0-indexed even)
    even_depths = depths[1::2]  # 2nd, 4th, 6th... (0-indexed odd)

    if len(odd_depths) == 0 or len(even_depths) == 0:
        return VettingCheckResult(
            name="odd_even_depth",
            passed=True,
            value=0.0,
            threshold=0.3,
            description="Need both odd and even transits",
        )

    median_odd = np.median(odd_depths)
    median_even = np.median(even_depths)
    
    if median_odd <= 0 or median_even <= 0:
        return VettingCheckResult(
            name="odd_even_depth",
            passed=True,
            value=0.0,
            threshold=0.3,
            description="Non-positive depths detected",
        )

    # Fractional difference
    frac_diff = abs(median_odd - median_even) / max(median_odd, median_even)
    
    # Threshold: >30% difference flags EB suspect
    threshold = 0.3
    passed = frac_diff < threshold
    
    return VettingCheckResult(
        name="odd_even_depth",
        passed=passed,
        value=float(frac_diff),
        threshold=threshold,
        description=f"Odd/even fractional depth difference: {frac_diff:.3f} (threshold: {threshold})",
    )


def duration_period_consistency(
    period: float,
    duration: float,
    depth_ppm: float,
    stellar_radius_rsun: Optional[float] = None,
) -> VettingCheckResult:
    """
    Duration-to-Period & Box-Shape Consistency Check.

    Planetary transits follow a well-defined duration-period relation
    for a given stellar density. Grazing eclipsing binaries have
    anomalously short durations (V-shaped) or inconsistent shapes.
    
    Parameters
    ----------
    period : float
        Orbital period [days]
    duration : float
        Transit duration [days]
    depth_ppm : float
        Transit depth [ppm]
    stellar_radius_rsun : float, optional
        Stellar radius [Rsun] for physical consistency check

    Returns
    -------
    VettingCheckResult
        passed=True if consistent with planetary transit geometry
    """
    if period <= 0 or duration <= 0:
        return VettingCheckResult(
            name="duration_period_consistency",
            passed=False,
            value=0.0,
            threshold=0.0,
            description="Invalid period or duration",
        )

    # Expected duration for a central transit around a Sun-like star
    # Using approximate relation: duration ~ (P/π) * (R_star/a) * sqrt(1-b^2)
    # For circular orbit: a ~ (P^2)^(1/3) in AU (for 1 Msun)
    # Simplified: duration/period ~ 0.01 - 0.1 for typical planets
    
    dur_per_ratio = duration / period
    
    # Physical limits: 
    # - Minimum duration for a grazing transit: ~0.1% of period (very short)
    # - Maximum for central transit around giant star: ~10% of period
    # - Typical hot Jupiter: ~1-3% of period
    min_ratio = 0.001
    max_ratio = 0.1
    
    # Also check for V-shape vs U-shape via depth/duration relation
    # Deep, short transits are more likely grazing EBs
    depth_frac = depth_ppm / 1e6
    
    passed = True
    reasons = []
    
    if dur_per_ratio < min_ratio:
        passed = False
        reasons.append(f"Duration/period ratio {dur_per_ratio:.4f} < {min_ratio} (grazing/V-shaped)")
    elif dur_per_ratio > max_ratio:
        passed = False
        reasons.append(f"Duration/period ratio {dur_per_ratio:.4f} > {max_ratio} (unphysically long)")
    
    # Flag very deep + short transits (grazing EB signature)
    if depth_frac > 0.01 and dur_per_ratio < 0.005:
        passed = False
        reasons.append(f"Deep ({depth_ppm:.0f} ppm) + short transit: grazing EB suspect")
    
    desc = "; ".join(reasons) if reasons else f"Duration/period ratio: {dur_per_ratio:.4f} (OK)"
    
    return VettingCheckResult(
        name="duration_period_consistency",
        passed=passed,
        value=float(dur_per_ratio),
        threshold=max_ratio,
        description=desc,
    )


def box_shape_consistency(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> VettingCheckResult:
    """
    Box-Shape Consistency Check: U-shaped (planetary) vs V-shaped (grazing EB).

    Fits both box and trapezoid models to the phase-folded transit
    and compares residuals. Planetary transits are more box-like
    (flat bottom); grazing EBs are V-shaped (no flat bottom).
    """
    if len(time) < 20:
        return VettingCheckResult(
            name="box_shape_consistency",
            passed=True,
            value=0.0,
            threshold=0.5,
            description="Insufficient data for shape analysis",
        )

    # Phase fold around the transit
    phase = ((time - t0) % period) / period
    # Also check secondary eclipse at phase 0.5
    phase_secondary = ((time - t0 - 0.5 * period) % period) / period
    
    # Primary transit window
    w = duration / period
    primary_mask = (phase < w) | (phase > 1 - w)
    # Secondary eclipse window
    secondary_mask = (phase_secondary < w) | (phase_secondary > 1 - w)
    
    # Get in-transit and out-of-transit fluxes
    in_primary = flux[primary_mask]
    out_flux = flux[~primary_mask & ~secondary_mask]
    
    if len(in_primary) < 5 or len(out_flux) < 10:
        return VettingCheckResult(
            name="box_shape_consistency",
            passed=True,
            value=0.0,
            threshold=0.5,
            description="Insufficient in/out transit points",
        )
    
    # Sort in-transit points by phase for shape analysis
    primary_phases = phase[primary_mask]
    sort_idx = np.argsort(primary_phases)
    in_phases = primary_phases[sort_idx]
    in_flux_sorted = in_primary[sort_idx]
    
    # Fit box model (constant in-transit)
    box_level = np.median(in_flux_sorted)
    box_residuals = in_flux_sorted - box_level
    box_rms = np.std(box_residuals)
    
    # Fit linear model (V-shape)
    if len(in_phases) > 2:
        # Normalize phases to [-0.5, 0.5] around mid-transit
        centered_phases = in_phases - 0.5
        # Linear fit
        A = np.vstack([centered_phases, np.ones_like(centered_phases)]).T
        try:
            slope, intercept = np.linalg.lstsq(A, in_flux_sorted, rcond=None)[0]
            v_model = slope * centered_phases + intercept
            v_residuals = in_flux_sorted - v_model
            v_rms = np.std(v_residuals)
        except np.linalg.LinAlgError:
            v_rms = box_rms
    else:
        v_rms = box_rms
    
    # Shape metric: ratio of V-fit RMS to box-fit RMS
    # If box is better (planetary), ratio > 1
    # If V-shape is better (grazing EB), ratio < 1
    if box_rms > 0:
        shape_ratio = v_rms / box_rms
    else:
        shape_ratio = 1.0
    
    # Threshold: box should fit better than V-shape for planetary transits
    # Allow some tolerance for noise
    threshold = 0.8
    passed = shape_ratio > threshold
    
    return VettingCheckResult(
        name="box_shape_consistency",
        passed=passed,
        value=float(shape_ratio),
        threshold=threshold,
        description=f"V-shape/box RMS ratio: {shape_ratio:.3f} (>{threshold} favors box/U-shape)",
    )


def momentum_dump_proximity(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    sector: Optional[int] = None,
    cadence_tolerance: float = 0.5,  # days
) -> VettingCheckResult:
    """
    Instrumental / Momentum Dump Proximity Filter.

    Flags transits that cluster near known TESS momentum dump cadences
    or sector gap edges. These are common sources of instrumental artifacts.
    
    Parameters
    ----------
    time : np.ndarray
        Time array [BJD]
    flux : np.ndarray
        Detrended flux array
    period : float
        Orbital period [days]
    t0 : float
        Reference mid-transit time [BJD]
    duration : float
        Transit duration [days]
    sector : int, optional
        TESS sector number for known dump cadences
    cadence_tolerance : float
        Time tolerance [days] for proximity flag
    
    Returns
    -------
    VettingCheckResult
        passed=True if no proximity to known artifacts
    """
    # Known momentum dump times per sector (approximate BJD)
    # In practice, these would come from TESS Data Release Notes
    # For now, use a heuristic: check if transits align with regular
    # spacecraft momentum dump cadence (~2.5 days for some sectors)
    
    # Generate all transit times in the observation window
    t_min, t_max = time[0], time[-1]
    transit_times = []
    epoch = 0
    while True:
        t_transit = t0 + epoch * period
        if t_transit > t_max:
            break
        if t_transit >= t_min:
            transit_times.append(t_transit)
        epoch += 1
    
    if not transit_times:
        return VettingCheckResult(
            name="momentum_dump_proximity",
            passed=True,
            value=0.0,
            threshold=cadence_tolerance,
            description="No transits in observation window",
        )
    
    # Check proximity to sector boundaries (gaps)
    # Sector gaps typically occur at regular intervals
    # Heuristic: flag if any transit is within tolerance of observation edges
    min_edge_dist = min(
        min(abs(tt - t_min) for tt in transit_times),
        min(abs(tt - t_max) for tt in transit_times),
    )
    
    # Check for regular spacing artifacts (momentum dumps every ~2.5 days in some sectors)
    # Compute transit times modulo ~2.5 days
    dump_cadence = 2.5  # days, approximate
    min_dump_dist = float('inf')
    for tt in transit_times:
        phase = tt % dump_cadence
        dist = min(phase, dump_cadence - phase)
        min_dump_dist = min(min_dump_dist, dist)
    
    # Also check for data gap proximity
    # Find large time gaps in the light curve
    dt = np.diff(time)
    median_dt = np.median(dt)
    gap_threshold = 5 * median_dt  # 5x median cadence = gap
    gap_starts = time[:-1][dt > gap_threshold]
    gap_ends = time[1:][dt > gap_threshold]
    
    min_gap_dist = float('inf')
    for tt in transit_times:
        for gs, ge in zip(gap_starts, gap_ends):
            dist = min(abs(tt - gs), abs(tt - ge))
            min_gap_dist = min(min_gap_dist, dist)
    
    closest_approach = min(min_edge_dist, min_dump_dist, min_gap_dist)
    passed = closest_approach > cadence_tolerance
    
    return VettingCheckResult(
        name="momentum_dump_proximity",
        passed=passed,
        value=float(closest_approach),
        threshold=cadence_tolerance,
        description=f"Closest approach to artifact: {closest_approach:.3f} days (tol: {cadence_tolerance})",
    )


def signal_to_pink_noise_ratio(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> VettingCheckResult:
    """
    Signal-to-Pink-Noise Ratio (SPNR) accounting for correlated red noise.

    Standard SNR underestimates uncertainties when red noise is present.
    Computes SPNR using time-averaging method (Pont et al. 2006) or
    wavelet-based likelihood (Carter & Winn 2009).
    
    Simplified implementation: compare RMS at transit duration timescale
    to white-noise expectation.
    """
    if len(time) < 50:
        return VettingCheckResult(
            name="signal_to_pink_noise",
            passed=True,
            value=0.0,
            threshold=5.0,
            description="Insufficient data for SPNR",
        )

    # Phase fold and isolate transit
    phase = ((time - t0) % period) / period
    in_transit = phase < (duration / period)
    out_transit = ~in_transit
    
    in_flux = flux[in_transit]
    out_flux = flux[out_transit]
    
    if len(in_flux) < 3 or len(out_flux) < 10:
        return VettingCheckResult(
            name="signal_to_pink_noise",
            passed=True,
            value=0.0,
            threshold=5.0,
            description="Insufficient in/out transit points",
        )
    
    # Transit depth
    depth = np.median(out_flux) - np.median(in_flux)
    
    # White noise estimate from out-of-transit scatter
    out_residuals = out_flux - np.median(out_flux)
    white_noise = np.std(out_residuals)
    
    # Red noise estimate: bin out-of-transit residuals at transit duration timescale
    # and measure excess variance
    cadence_d = np.median(np.diff(time))
    bin_size = max(1, int(duration / cadence_d))
    
    if bin_size > 1 and len(out_residuals) > bin_size * 2:
        # Bin the out-of-transit residuals
        n_bins = len(out_residuals) // bin_size
        binned = out_residuals[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
        binned_std = np.std(binned)
        
        # White noise prediction for binned data
        white_prediction = white_noise / np.sqrt(bin_size)
        
        # Red noise factor (beta)
        beta = binned_std / max(white_prediction, 1e-12)
        beta = max(beta, 1.0)  # At least white noise
    else:
        beta = 1.0
    
    # Effective noise including red noise
    eff_noise = white_noise * beta / np.sqrt(len(in_flux))
    
    spnr = abs(depth) / max(eff_noise, 1e-12)
    
    # Threshold: SPNR > 7 for confident detection (vs typical SNR > 7-10)
    threshold = 7.0
    passed = spnr >= threshold
    
    return VettingCheckResult(
        name="signal_to_pink_noise",
        passed=passed,
        value=float(spnr),
        threshold=threshold,
        description=f"SPNR: {spnr:.2f} (beta={beta:.2f}, threshold: {threshold})",
    )


def secondary_eclipse_check(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    depth_ppm: float,
) -> VettingCheckResult:
    """
    Secondary Eclipse Check.

    Eclipsing binaries often show secondary eclipses at phase 0.5.
    Planetary transits should not have significant secondary eclipses
    (except for very hot Jupiters with thermal emission, which are
    much shallower than primary).
    """
    if len(time) < 20:
        return VettingCheckResult(
            name="secondary_eclipse",
            passed=True,
            value=0.0,
            threshold=0.5,
            description="Insufficient data for secondary eclipse check",
        )

    # Phase fold and check phase 0.5
    phase = ((time - t0) % period) / period
    w = duration / period
    
    # Primary transit mask
    primary_mask = (phase < w) | (phase > 1 - w)
    # Secondary eclipse mask (phase 0.5)
    secondary_mask = (phase > 0.5 - w) & (phase < 0.5 + w)
    # Out of transit
    out_mask = ~primary_mask & ~secondary_mask
    
    if secondary_mask.sum() < 3 or out_mask.sum() < 10:
        return VettingCheckResult(
            name="secondary_eclipse",
            passed=True,
            value=0.0,
            threshold=0.5,
            description="Insufficient points at secondary phase",
        )
    
    secondary_flux = flux[secondary_mask]
    out_flux = flux[out_mask]
    
    secondary_depth = np.median(out_flux) - np.median(secondary_flux)
    primary_depth_frac = depth_ppm / 1e6
    
    if primary_depth_frac <= 0:
        return VettingCheckResult(
            name="secondary_eclipse",
            passed=True,
            value=0.0,
            threshold=0.5,
            description="Non-positive primary depth",
        )
    
    # Secondary-to-primary depth ratio
    # For EBs: ratio ~ 0.1 - 1.0 (depending on temperature ratio)
    # For planets: ratio ~ 0 (optical) to ~1e-4 (thermal emission in IR)
    ratio = secondary_depth / primary_depth_frac if primary_depth_frac > 0 else 0
    
    # Threshold: flag if secondary > 10% of primary (EB suspect)
    # Allow up to 50% for grazing EBs with similar depths
    threshold = 0.1
    passed = ratio < threshold
    
    return VettingCheckResult(
        name="secondary_eclipse",
        passed=passed,
        value=float(ratio),
        threshold=threshold,
        description=f"Secondary/primary depth ratio: {ratio:.3f} (threshold: {threshold})",
    )


def run_all_checks(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    depth_ppm: float,
    sector: Optional[int] = None,
    stellar_radius_rsun: Optional[float] = None,
) -> List[VettingCheckResult]:
    """
    Run all vetting checks and return list of results.
    """
    checks = [
        odd_even_depth_test(time, flux, period, t0, duration),
        duration_period_consistency(period, duration, depth_ppm, stellar_radius_rsun),
        box_shape_consistency(time, flux, period, t0, duration),
        momentum_dump_proximity(time, flux, period, t0, duration, sector),
        signal_to_pink_noise_ratio(time, flux, period, t0, duration),
        secondary_eclipse_check(time, flux, period, t0, duration, depth_ppm),
    ]
    
    logger.info("Completed %d vetting checks", len(checks))
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        logger.debug("  %s: %s - %s", c.name, status, c.description)
    
    return checks