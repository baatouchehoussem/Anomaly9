"""
Anomaly9 Dual-Core Detector
===========================
Engine 1: Box Least Squares (BLS) transit periodogram (hand-rolled numpy,
          box model with optimal depth/duration estimation).
Engine 2: Unsupervised anomaly detection: sliding-window robust Z-score +
          Isolation Forest scoring over (flux, z) features; contiguous
          anomalous points are clustered into physical event candidates
          (single transits, flares, dips, instrument glitches).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger("anomaly9.detector")

__all__ = [
    "BLSResult",
    "AnomalyEvent",
    "DetectionReport",
    "run_bls",
    "sliding_zscore",
    "run_isolation_forest",
    "cluster_anomalies",
    "run_dual_detector",
]


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class BLSResult:
    period_d: float            # best-fit period [days]
    depth_ppm: float           # transit depth [ppm, negative = dip]
    duration_d: float          # best-fit duration [days]
    t0_bjd: float              # mid-transit time [BJD]
    snr: float                 # signal-to-noise of the box
    in_transit_frac: float     # fraction of cadences in transit
    chi2: float                # chi^2 of the box model
    # periodogram arrays
    periods: np.ndarray = field(repr=False, default=None)
    powers: np.ndarray = field(repr=False, default=None)
    depths: np.ndarray = field(repr=False, default=None)


@dataclass
class AnomalyEvent:
    t_start: float             # BJD
    t_end: float               # BJD
    t_mid: float               # BJD
    duration_d: float
    depth_ppm: float           # negative for dips, positive for flares
    kind: str                  # 'dip' | 'flare' | 'mixed'
    min_zscore: float
    n_points: int


@dataclass
class DetectionReport:
    bls_best: Optional[BLSResult]
    bls_top: List[BLSResult]
    zscore: np.ndarray
    iforest_scores: np.ndarray
    iforest_flags: np.ndarray
    events: List[AnomalyEvent]


# --------------------------------------------------------------------------
# Engine 1: Box Least Squares
# --------------------------------------------------------------------------

def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    period_min_d: float = 0.5,
    period_max_d: float = 10.0,
    period_step_d: float = 0.001,
    durations_d: Optional[np.ndarray] = None,
    top_n: int = 5,
) -> List[BLSResult]:
    """Hand-rolled numpy BLS (Kovacs+2002 power statistic).

    For each trial period the time series is phase-folded; for each trial
    duration the box depth is the in-transit minus out-of-transit mean of the
    detrended flux and the Kovacs+2002 power is the depth in sigma, weighted
    by the harmonic mean of in/out sample counts. Returns the top-N peaks as
    BLSResult objects (first is the global best; periodogram arrays attached).
    """
    n = time.size
    if n < 30:
        raise ValueError(f"BLS requires >= 30 inliers, got {n}.")

    if durations_d is None:
        durations_d = np.unique(np.clip(np.logspace(np.log10(0.02), np.log10(0.25), 24), 0.01, 0.5))

    periods = np.arange(period_min_d, period_max_d, period_step_d)
    if periods.size < 10:
        periods = np.linspace(period_min_d, period_max_d, 500)

    median = np.median(flux)
    mad = np.median(np.abs(flux - median)) * 1.4826
    sigma = max(mad, 1e-8)
    y = flux - median                      # baseline-shifted

    best_power = np.zeros(periods.size)
    best_depth = np.zeros(periods.size)
    best_dur = np.zeros(periods.size)
    best_t0 = np.zeros(periods.size)
    best_frac = np.zeros(periods.size)
    best_chi2 = np.zeros(periods.size)

    for k, P in enumerate(periods):
        phase = ((time - time[0]) % P) / P
        max_pow = -np.inf
        max_idx = -1
        for di, dur in enumerate(durations_d):
            wfrac = dur / P
            if wfrac >= 0.5:
                continue
            in_mask = phase < wfrac
            n_in = int(in_mask.sum())
            n_out = n - n_in
            if n_in < 3 or n_out < 10:
                continue
            mean_in = y[in_mask].mean()
            mean_out = y[~in_mask].mean()
            # Kovács+2002 style power: dip depth in sigma, weighted by the
            # harmonic mean of in/out sample sizes.
            power = (mean_out - mean_in) / sigma * np.sqrt(n_in * n_out / n)
            if power > max_pow:
                max_pow = power
                max_idx = di

        if max_idx >= 0 and np.isfinite(max_pow):
            wfrac = durations_d[max_idx] / P
            in_mask = phase < wfrac
            mean_in = y[in_mask].mean()
            mean_out = y[~in_mask].mean()
            n_in = int(in_mask.sum())
            n_out = n - n_in
            best_power[k] = max_pow
            best_depth[k] = mean_in - mean_out
            best_dur[k] = durations_d[max_idx]
            best_frac[k] = wfrac
            # chi^2 reduction of the box model vs a flat line (in units of sigma)
            best_chi2[k] = (n_in * n_out / n) * ((mean_out - mean_in) / sigma) ** 2
            mid_phase = float(np.median(phase[in_mask]))
            best_t0[k] = time[0] + mid_phase * P

    order = np.argsort(-best_power)
    results: List[BLSResult] = []
    seen = 0
    for k in order:
        if best_power[k] <= 0 or not np.isfinite(best_power[k]):
            continue
        r = BLSResult(
            period_d=float(periods[k]),
            depth_ppm=float(best_depth[k] * 1e6),
            duration_d=float(best_dur[k]),
            t0_bjd=float(best_t0[k]),
            snr=float(best_power[k]),
            in_transit_frac=float(best_frac[k]),
            chi2=float(best_chi2[k]),
            periods=periods,
            powers=best_power,
            depths=best_depth,
        )
        results.append(r)
        seen += 1
        if seen >= top_n:
            break
    return results


# --------------------------------------------------------------------------
# Engine 2: Unsupervised anomaly detection
# --------------------------------------------------------------------------

def sliding_zscore(flux: np.ndarray, window: int = 25) -> np.ndarray:
    """Robust rolling Z-score: (flux - rolling_median) / rolling_MAD.

    Vectorized with a sliding window view; edge windows shrink gracefully.
    """
    n = flux.size
    if n < window:
        med = np.median(flux)
        mad = np.median(np.abs(flux - med)) * 1.4826
        return np.clip((flux - med) / max(mad, 1e-12), -20, 20)

    half = window // 2
    padded = np.pad(flux, half, mode="edge")
    wins = np.lib.stride_tricks.sliding_window_view(padded, window)
    med = np.median(wins, axis=1)
    mad = np.median(np.abs(wins - med[:, None]), axis=1) * 1.4826
    z = (flux - med) / np.maximum(mad, 1e-12)
    return np.clip(z, -20, 20)


def run_isolation_forest(
    time: np.ndarray,
    flux: np.ndarray,
    zscore: np.ndarray,
    contamination: float = 0.01,
    random_state: int = 42,
):
    """Isolation Forest over (flux, z) features.

    Returns (scores, flags) where scores = decision_function (negative =
    anomalous) and flags = boolean mask of detected anomalies.
    """
    from sklearn.ensemble import IsolationForest

    X = np.column_stack([flux - np.median(flux), zscore])
    X = np.nan_to_num(X, nan=0.0, posinf=20.0, neginf=-20.0)
    if X.shape[0] < 50:
        # too few points for a meaningful forest; fall back to z-threshold only
        scores = -np.abs(zscore)
        flags = np.abs(zscore) > 4.0
        return scores, flags

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X)
    scores = iso.decision_function(X)
    # combine with robust z-gate: an anomaly must be either a strong forest
    # outlier OR a deep robust-z dip/bump (dips dominate astrophysical events)
    threshold = float(np.percentile(scores, 100 * contamination))
    flags = (scores < threshold) | (zscore < -4.0) | (zscore > 6.0)
    return scores, flags


def cluster_anomalies(
    time: np.ndarray,
    flux: np.ndarray,
    zscore: np.ndarray,
    flags: np.ndarray,
    max_gap: int = 3,
) -> List[AnomalyEvent]:
    """Group contiguous flagged indices (allowing small gaps) into events."""
    events: List[AnomalyEvent] = []
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return events

    groups: List[np.ndarray] = []
    current = [idx[0]]
    for i in idx[1:]:
        if i - current[-1] <= max_gap + 1:
            current.append(i)
        else:
            groups.append(np.array(current))
            current = [i]
    groups.append(np.array(current))

    baseline = np.median(flux)
    for g in groups:
        if g.size < 2:
            continue
        seg = flux[g]
        delta = seg - baseline
        depth = float(np.min(delta) if np.min(delta) < 0 else np.max(delta))
        if depth <= -1e-6:
            kind = "dip"
        elif depth > 1e-6:
            kind = "flare"
        else:
            continue
        t_mid = float(np.median(time[g]))
        events.append(
            AnomalyEvent(
                t_start=float(time[g[0]]),
                t_end=float(time[g[-1]]),
                t_mid=t_mid,
                duration_d=float(time[g[-1]] - time[g[0]] + (time[1] - time[0]) if time.size > 1 else 0.0),
                depth_ppm=depth * 1e6,
                kind=kind,
                min_zscore=float(np.min(zscore[g])),
                n_points=int(g.size),
            )
        )
    events.sort(key=lambda e: e.t_mid)
    return events


# --------------------------------------------------------------------------
# Dual-engine driver
# --------------------------------------------------------------------------

def run_dual_detector(
    data,
    period_min_d: float = 0.5,
    period_max_d: float = 10.0,
    period_step_d: float = 0.002,
    contamination: float = 0.01,
    z_window: int = 25,
    bls_top_n: int = 5,
) -> DetectionReport:
    """Run both engines on a LightCurveData and return a DetectionReport."""
    t, f = data.time, data.flux

    logger.info("Engine 1: BLS periodogram over [%.2f, %.2f] d ...", period_min_d, period_max_d)
    bls_top = run_bls(
        t, f,
        period_min_d=period_min_d,
        period_max_d=period_max_d,
        period_step_d=period_step_d,
        top_n=bls_top_n,
    )
    bls_best = bls_top[0] if bls_top else None
    if bls_best is not None:
        logger.info(
            "BLS best: P=%.4f d, depth=%.1f ppm, dur=%.3f d, t0=%.4f, SNR=%.1f",
            bls_best.period_d, bls_best.depth_ppm, bls_best.duration_d, bls_best.t0_bjd, bls_best.snr,
        )
    else:
        logger.warning("BLS found no positive peaks.")

    logger.info("Engine 2: Isolation Forest + robust Z-score ...")
    z = sliding_zscore(f, window=z_window)
    scores, flags = run_isolation_forest(t, f, z, contamination=contamination)
    events = cluster_anomalies(t, f, z, flags)
    logger.info("Detected %d anomaly event(s).", len(events))

    return DetectionReport(
        bls_best=bls_best,
        bls_top=bls_top,
        zscore=z,
        iforest_scores=scores,
        iforest_flags=flags,
        events=events,
    )
