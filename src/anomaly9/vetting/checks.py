"""
Anomaly9 Vetting Checks
=======================
Modular false-positive vetting checks for TESS light curve anomalies.

Implements standard astrophysical false-positive diagnostics:
- Odd/Even Transit Depth Test (eclipsing binary flag)
- Duration-to-Period & Box-Shape Consistency (grazing eclipse vs planetary)
- Instrumental / Momentum Dump Proximity Filter
- Signal-to-Pink-Noise Ratio (SPNR) for correlated red noise
- Secondary Eclipse Check (self-luminous companion)
- Transit Shape Boxiness (U-shape vs V-shape)
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
    # Ensure positive depths (transits are dips)
    depths = np.abs(depths)
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
    stellar_density_gcc: Optional[float] = None,
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
    stellar_density_gcc : float, optional
        Stellar density [g/cm^3] for physical duration prediction
        using Seager & Mallen-Ornelas (2003) relation.

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

    dur_per_ratio = duration / period
    
    # Physical constants
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    
    if stellar_density_gcc is not None and stellar_density_gcc > 0:
        # Use physical stellar density to compute expected duration
        # From Seager & Mallen-Ornelas (2003):
        # a/R* = (G * M* * P^2 / (4*pi^2 * R*^3))^(1/3) = (P^2 * G * rho* / (3*pi))^(1/3)
        # For central transit (b=0): duration ~ (P/pi) * (R*/a)
        # duration/period ~ (1/pi) * (R*/a) = 1 / (pi * a/R*)
        
        # Convert period to seconds
        P_sec = period * 86400.0
        # Convert density from g/cm^3 to kg/m^3
        rho_kgm3 = stellar_density_gcc * 1000.0
        
        # Compute a/R* using the density relation
        # a/R* = (P^2 * G * rho / (3*pi))^(1/3)
        a_over_Rstar = (P_sec**2 * G * rho_kgm3 / (3.0 * np.pi))**(1.0/3.0)
        
        # Expected duration/period for central transit (b=0)
        # duration/period = (1/pi) * (R*/a) = 1 / (pi * a/R*)
        expected_dur_per_ratio = 1.0 / (np.pi * a_over_Rstar)
        
        # Allow factor of 2 variation (0.5x to 2x expected)
        min_expected = 0.5 * expected_dur_per_ratio
        max_expected = 2.0 * expected_dur_per_ratio
        
        passed = True
        reasons = []
        
        if dur_per_ratio < min_expected:
            passed = False
            reasons.append(f"Duration/period {dur_per_ratio:.4f} < {min_expected:.4f} (0.5x expected from density rho*={stellar_density_gcc:.2f} g/cm3)")
        elif dur_per_ratio > max_expected:
            passed = False
            reasons.append(f"Duration/period {dur_per_ratio:.4f} > {max_expected:.4f} (2x expected from density rho*={stellar_density_gcc:.2f} g/cm3)")
        
        # Also check for V-shape vs U-shape via depth/duration relation
        depth_frac = depth_ppm / 1e6
        if depth_frac > 0.01 and dur_per_ratio < 0.005:
            passed = False
            reasons.append(f"Deep ({depth_ppm:.0f} ppm) + short transit: grazing EB suspect")
        
        desc = "; ".join(reasons) if reasons else f"Duration/period: {dur_per_ratio:.4f} (expected {expected_dur_per_ratio:.4f} +/- factor 2 from rho*={stellar_density_gcc:.2f} g/cm3)"
        
        return VettingCheckResult(
            name="duration_period_consistency",
            passed=passed,
            value=float(dur_per_ratio),
            threshold=float(max_expected),
            description=desc,
        )
    
# Fallback to heuristic if no density provided
    # Expected duration for a central transit around a Sun-like star
    # Using approximate relation: duration ~ (P/pi) * (R_star/a) * sqrt(1-b^2)
    # For circular orbit: a ~ (P^2)^(1/3) in AU (for 1 Msun)
    # Simplified: duration/period ~ 0.01 - 0.1 for typical planets
    
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
    
    desc = "; ".join(reasons) if reasons else f"Duration/period ratio: {dur_per_ratio:.4f} (OK, heuristic limits {min_ratio}-{max_ratio})"
    
    return VettingCheckResult(
        name="duration_period_consistency",
        passed=passed,
        value=float(dur_per_ratio),
        threshold=max_ratio,
        description=desc,
    )


def transit_shape_boxiness(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> VettingCheckResult:
    """
    Transit Shape Boxiness Test: U-shaped (planetary) vs V-shaped (grazing EB).

    Fits a trapezoid model (with ingress/egress) and a box model to the
    phase-folded transit. Planetary transits have flat bottoms (box-like);
    grazing EBs are V-shaped with no flat bottom.
    
    Uses the ratio of trapezoid-fit residuals to box-fit residuals.
    For planetary transits, box model fits well (ratio ~1).
    For grazing EBs, trapezoid fits much better (ratio < 0.7).
    """
    if len(time) < 20:
        return VettingCheckResult(
            name="transit_shape_boxiness",
            passed=True,
            value=1.0,
            threshold=0.7,
            description="Insufficient data for shape analysis",
        )

    # Phase fold around the transit
    phase = ((time - t0) % period) / period
    
    # Primary transit window
    w = duration / period
    primary_mask = (phase < w) | (phase > 1 - w)
    
    # Get in-transit and out-of-transit fluxes
    in_primary = flux[primary_mask]
    out_flux = flux[~primary_mask]
    
    if len(in_primary) < 5 or len(out_flux) < 10:
        return VettingCheckResult(
            name="transit_shape_boxiness",
            passed=True,
            value=1.0,
            threshold=0.7,
            description="Insufficient in/out transit points",
        )
    
    # Sort in-transit points by phase
    primary_phases = phase[primary_mask]
    sort_idx = np.argsort(primary_phases)
    in_phases = primary_phases[sort_idx]
    in_flux_sorted = in_primary[sort_idx]
    
    # Normalize phases to [0, 1] across transit
    in_phases_norm = (in_phases - in_phases[0]) / (in_phases[-1] - in_phases[0] + 1e-10)
    
    # Fit box model (constant in-transit)
    box_level = np.median(in_flux_sorted)
    box_residuals = in_flux_sorted - box_level
    box_rms = np.std(box_residuals)
    
    # Fit trapezoid model (flat bottom with linear ingress/egress)
    # Trapezoid has 4 parameters: level, ingress_slope, egress_slope, mid_transit_fraction
    # Simplified: fit piecewise linear with flat middle
    if len(in_phases) > 4:
        try:
            # Split into ingress, flat, egress (25%/50%/25%)
            n = len(in_phases)
            ingress_end = n // 4
            egress_start = 3 * n // 4
            
            # Ingress: linear fit to first 25%
            if ingress_end > 1:
                A_ing = np.vstack([in_phases_norm[:ingress_end], np.ones(ingress_end)]).T
                slope_ing, intercept_ing = np.linalg.lstsq(A_ing, in_flux_sorted[:ingress_end], rcond=None)[0]
            
            # Egress: linear fit to last 25%
            if n - egress_start > 1:
                A_eg = np.vstack([in_phases_norm[egress_start:], np.ones(n - egress_start)]).T
                slope_eg, intercept_eg = np.linalg.lstsq(A_eg, in_flux_sorted[egress_start:], rcond=None)[0]
            
            # Flat bottom: median of middle 50%
            flat_level = np.median(in_flux_sorted[ingress_end:egress_start])
            
            # Build trapezoid model
            trap_model = np.zeros_like(in_flux_sorted)
            trap_model[:ingress_end] = slope_ing * in_phases_norm[:ingress_end] + intercept_ing
            trap_model[ingress_end:egress_start] = flat_level
            trap_model[egress_start:] = slope_eg * in_phases_norm[egress_start:] + intercept_eg
            
            trap_residuals = in_flux_sorted - trap_model
            trap_rms = np.std(trap_residuals)
        except (np.linalg.LinAlgError, IndexError):
            trap_rms = box_rms
    else:
        trap_rms = box_rms
    
    # Shape metric: ratio of box RMS to trapezoid RMS
    # If box fits as well as trapezoid (planetary), ratio ~1
    # If trapezoid fits much better (grazing EB), ratio > 1.3
    if trap_rms > 0:
        shape_ratio = box_rms / trap_rms
    else:
        shape_ratio = 1.0
    
    # Threshold: box should fit nearly as well as trapezoid for planetary transits
    threshold = 0.7
    passed = shape_ratio >= threshold
    
    return VettingCheckResult(
        name="transit_shape_boxiness",
        passed=passed,
        value=float(shape_ratio),
        threshold=threshold,
        description=f"Box/trapezoid RMS ratio: {shape_ratio:.3f} (>{threshold} favors U-shape/planetary)",
    )


def momentum_dump_proximity(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    sector: Optional[int] = None,
    cadence_tolerance: float = 0.2,  # days - tighter tolerance
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
    
    # Use the minimum of all distances
    closest_approach = min(min_edge_dist, min_dump_dist, min_gap_dist)
    
    # Stricter tolerance: only flag if transit is VERY close to artifact
    passed = closest_approach > cadence_tolerance
    
    # Determine which artifact is closest
    artifact_type = "none"
    if min_edge_dist == closest_approach:
        artifact_type = "sector edge"
    elif min_dump_dist == closest_approach:
        artifact_type = "momentum dump"
    elif min_gap_dist == closest_approach:
        artifact_type = "data gap"
    
    return VettingCheckResult(
        name="momentum_dump_proximity",
        passed=passed,
        value=float(closest_approach),
        threshold=cadence_tolerance,
        description=f"Closest approach to {artifact_type}: {closest_approach:.3f} days (tol: {cadence_tolerance})",
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
    
    # Transit depth (positive for dips)
    depth = abs(np.median(out_flux) - np.median(in_flux))
    
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
    
    spnr = depth / max(eff_noise, 1e-12)
    
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
            threshold=0.1,
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
            threshold=0.1,
            description="Insufficient points at secondary phase",
        )
    
    secondary_flux = flux[secondary_mask]
    out_flux = flux[out_mask]
    
    # Depths are positive for dips
    secondary_depth = abs(np.median(out_flux) - np.median(secondary_flux))
    primary_depth_frac = depth_ppm / 1e6
    
    if primary_depth_frac <= 0:
        return VettingCheckResult(
            name="secondary_eclipse",
            passed=True,
            value=0.0,
            threshold=0.1,
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
    stellar_density_gcc: Optional[float] = None,
) -> List[VettingCheckResult]:
    """
    Run all vetting checks and return list of results.
    """
    checks = [
        odd_even_depth_test(time, flux, period, t0, duration),
        duration_period_consistency(period, duration, depth_ppm, stellar_radius_rsun, stellar_density_gcc),
        transit_shape_boxiness(time, flux, period, t0, duration),
        momentum_dump_proximity(time, flux, period, t0, duration, sector),
        signal_to_pink_noise_ratio(time, flux, period, t0, duration),
        secondary_eclipse_check(time, flux, period, t0, duration, depth_ppm),
    ]
    
    logger.info("Completed %d vetting checks", len(checks))
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        logger.debug("  %s: %s - %s", c.name, status, c.description)
    
    return checks
