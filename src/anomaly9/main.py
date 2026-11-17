"""
Anomaly9 CLI & Diagnostic Plotter
==================================
CLI entry point for Anomaly9 Phase 1. Orchestrates the MAST data pipeline,
runs the dual-core detector (BLS + Isolation Forest), prints structured
astrophysical metrics, and saves a publication-grade 3-panel diagnostic PNG.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np

from anomaly9.pipeline import ingest_mast_lightcurve, LightCurveData
from anomaly9.detector import run_dual_detector, DetectionReport, BLSResult

logger = logging.getLogger("anomaly9")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        level=level,
    )


def print_report(data: LightCurveData, report: DetectionReport):
    print("=" * 72)
    print(f" ANOMALY9 DETECTION REPORT: {data.target} ({data.mission})")
    print("=" * 72)
    print(f" Light Curve Points : {data.n_points} (cleaned, {data.n_removed} outliers removed)")
    print(f" Time Span          : {data.duration_days:.2f} days")
    print(f" Median Cadence     : {data.cadence_s:.1f} seconds")
    print(f" Sector / Quarter   : {data.metadata.get('sector') or data.metadata.get('quarter') or 'N/A'}")
    print("-" * 72)

    print(" [1] Box Least Squares (BLS) Transit Periodogram (Top Candidates):")
    if report.bls_top:
        print(f"   {'Rank':<5} {'Period [d]':<12} {'Depth [ppm]':<14} {'Duration [d]':<14} {'SNR':<8}")
        print("   " + "-" * 60)
        for i, b in enumerate(report.bls_top[:5], 1):
            marker = " *" if i == 1 else ""
            print(f"   {i:<5} {b.period_d:<12.5f} {b.depth_ppm:<14.1f} {b.duration_d:<14.4f} {b.snr:<8.1f}{marker}")
        if report.bls_best:
            b = report.bls_best
            print(f"\n   -> Best-fit Ephemeris: P = {b.period_d:.6f} d, t0 = {b.t0_bjd:.4f} BJD, Depth = {b.depth_ppm:.1f} ppm")
    else:
        print("   No significant periodic transit signals detected.")

    print("-" * 72)
    print(f" [2] Unsupervised Anomaly Detector (Isolation Forest + Z-score):")
    print(f"   Flagged Anomalous Cadences : {int(report.iforest_flags.sum())}")
    print(f"   Clustered Astrophysical Events : {len(report.events)}")
    if report.events:
        print(f"   {'Kind':<8} {'T_mid [BJD]':<14} {'Duration [h]':<14} {'Depth [ppm]':<14} {'Min Z':<8}")
        print("   " + "-" * 66)
        for ev in report.events[:10]:
            dur_h = ev.duration_d * 24.0
            print(f"   {ev.kind:<8} {ev.t_mid:<14.4f} {dur_h:<14.2f} {ev.depth_ppm:<14.1f} {ev.min_zscore:<8.2f}")
    print("=" * 72)


def generate_diagnostic_plot(data: LightCurveData, report: DetectionReport, output_path: Path):
    """Save a publication-grade 3-panel diagnostic PNG."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False, constrained_layout=True)

    t = data.time
    f_raw = data.raw_flux
    f_det = data.flux
    z = report.zscore
    flags = report.iforest_flags

    # --- Panel 1: Raw Flux & Trend Fit ---
    ax1 = axes[0]
    ax1.scatter(t, f_raw, s=3, c="#4c72b0", alpha=0.6, label="Cleaned SAP/PDCSAP Flux")
    # overlay Savitzky-Golay trend
    trend = f_raw / f_det
    ax1.plot(t, trend, c="#c44e52", lw=1.5, label="Savitzky-Golay Trend")
    ax1.set_ylabel("Normalized Flux", fontsize=11, fontweight="bold")
    ax1.set_title(f"Anomaly9 Diagnostic: {data.target} ({data.mission})", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, ls="--", alpha=0.5)

    # --- Panel 2: Detrended Flux & BLS Transit Model ---
    ax2 = axes[1]
    ax2.scatter(t, f_det, s=3, c="#555555", alpha=0.5, label="Detrended Flux")
    # Highlight anomaly events
    if report.events:
        for ev in report.events:
            mask = (t >= ev.t_start) & (t <= ev.t_end)
            if mask.any():
                ax2.scatter(t[mask], f_det[mask], s=12, c="#d95f02", zorder=5, label=f"Anomaly ({ev.kind})" if ev == report.events[0] else "")

    # Overlay BLS best-fit model if available
    if report.bls_best is not None:
        b = report.bls_best
        phase = ((t - data.time[0]) % b.period_d) / b.period_d
        # sort by phase for clean model line
        s_idx = np.argsort(phase)
        # Construct model box
        model_flux = np.ones_like(t)
        in_transit = (phase % 1.0) < (b.duration_d / b.period_d)
        model_flux[in_transit] += (b.depth_ppm / 1e6)  # depth is negative for dips
        ax2.scatter(t[s_idx], model_flux[s_idx], s=1, c="#1b9e77", zorder=4, label=f"BLS Best P={b.period_d:.4f}d")

    ax2.set_ylabel("Detrended Flux", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, ls="--", alpha=0.5)

    # --- Panel 3: BLS Periodogram & Isolation Forest Anomaly Scores ---
    ax3 = axes[2]
    if report.bls_best is not None and report.bls_best.periods is not None:
        p_arr = report.bls_best.periods
        pow_arr = report.bls_best.powers
        ax3_twin = ax3.twinx()
        ax3_twin.plot(p_arr, pow_arr, c="#7570b3", lw=1.0, label="BLS Power")
        ax3_twin.set_ylabel("BLS Power (SNR)", color="#7570b3", fontsize=10, fontweight="bold")
        ax3_twin.tick_params(axis="y", labelcolor="#7570b3")
        ax3_twin.set_ylim(bottom=0)

    # Plot robust Z-score & anomaly markers
    ax3.plot(t, z, c="#333333", lw=0.8, alpha=0.7, label="Robust Rolling Z-score")
    if flags.any():
        ax3.scatter(t[flags], z[flags], s=10, c="#e7298a", zorder=5, label="Isolation Forest Flag")
    ax3.axhline(-4.0, c="#d95f02", ls=":", lw=1.2, label="Z = -4 Threshold")
    ax3.set_ylabel("Robust Z-Score", fontsize=11, fontweight="bold")
    ax3.set_xlabel(f"Time [BJD - {data.time[0]:.2f}]", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax3.grid(True, ls="--", alpha=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved high-resolution 3-panel diagnostic plot to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly9 Phase 1: NASA MAST Ingestion, Detrending & Dual Detection Engine"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="TIC 25155310",
        help="Target identifier (TIC ID for TESS, KIC or target name for Kepler). Default: TIC 25155310",
    )
    parser.add_argument(
        "--mission",
        type=str,
        default="TESS",
        choices=["TESS", "Kepler"],
        help="Mission name ('TESS' or 'Kepler'). Default: TESS",
    )
    parser.add_argument(
        "--sector",
        type=int,
        default=None,
        help="Specific sector / quarter number to download. Default: shortest available.",
    )
    parser.add_argument(
        "--period-min",
        type=float,
        default=0.5,
        help="Minimum trial period for BLS search [days]. Default: 0.5",
    )
    parser.add_argument(
        "--period-max",
        type=float,
        default=10.0,
        help="Maximum trial period for BLS search [days]. Default: 10.0",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Isolation Forest contamination rate. Default: 0.01",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anomaly9_diagnostic.png",
        help="Path to save the 3-panel diagnostic plot PNG. Default: anomaly9_diagnostic.png",
    )
    parser.add_argument(
        "--fits-file",
        type=str,
        default=None,
        help="Optional path to a local FITS file to load directly (bypasses MAST query/download).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        logger.info("Starting Anomaly9 analysis pipeline for target %r", args.target)
        data = ingest_mast_lightcurve(
            target=args.target,
            mission=args.mission,
            sector=args.sector,
            fits_file=args.fits_file,
        )

        report = run_dual_detector(
            data=data,
            period_min_d=args.period_min,
            period_max_d=args.period_max,
            contamination=args.contamination,
        )

        print_report(data, report)

        out_path = Path(args.output).resolve()
        generate_diagnostic_plot(data, report, out_path)
        print(f"\nDiagnostic plot successfully saved to: {out_path}\n")

    except Exception as e:
        logger.exception("Anomaly9 pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
