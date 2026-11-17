"""
Anomaly9 Batch CLI & Multi-Target Pipeline
===========================================
Command-line interface for running the full Anomaly9 pipeline on multiple targets
with vetting, ranking, and diagnostic output.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anomaly9.pipeline import ingest_mast_lightcurve, LightCurveData
from anomaly9.detector import run_dual_detector, DetectionReport, BLSResult
from anomaly9.vetting import vet_candidate, VettingReport

logger = logging.getLogger("anomaly9.cli")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        level=level,
    )


def parse_catalog(catalog_path: Path) -> List[str]:
    """Parse a catalog file (newline-separated or CSV with 'tic_id' column)."""
    tic_ids = []
    with open(catalog_path, "r") as f:
        # Try CSV first
        content = f.read().strip()
        if not content:
            return tic_ids
        
        lines = content.splitlines()
        # Check if first line looks like a header
        if lines and ("tic" in lines[0].lower() or "id" in lines[0].lower()):
            # CSV with header
            import io
            f.seek(0)
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                for key in row:
                    if "tic" in key.lower() or "id" in key.lower():
                        val = row[key].strip()
                        if val:
                            tic_ids.append(val)
                        break
        else:
            # Newline-separated
            for line in lines:
                val = line.strip()
                if val and not val.startswith("#"):
                    tic_ids.append(val)
    return tic_ids


def process_single_target(
    tic_id: str,
    mission: str,
    sectors: Optional[List[int]],
    period_min: float,
    period_max: float,
    period_step: float,
    contamination: float,
    z_window: int,
    bls_top_n: int,
    output_dir: Path,
    diagnostic_dir: Path,
    cache_dir: Path,
    rate_limit_delay: float = 1.0,
) -> Optional[VettingReport]:
    """
    Process a single target through the full pipeline.
    Returns VettingReport or None on failure.
    """
    try:
        logger.info("Processing %s (mission=%s)", tic_id, mission)
        
        # Try each sector if specified, otherwise let pipeline choose
        data: Optional[LightCurveData] = None
        last_error = None
        
        sector_list = sectors if sectors else [None]
        
        for sector in sector_list:
            try:
                data = ingest_mast_lightcurve(
                    target=tic_id,
                    mission=mission,
                    sector=sector,
                    window_savgol=120,
                    poly_order=3,
                    sigma_clip=6.0,
                )
                if data.n_points > 30:
                    break
            except Exception as e:
                last_error = e
                logger.warning("Failed to ingest %s sector %s: %s", tic_id, sector, e)
                continue
        
        if data is None or data.n_points < 30:
            logger.error("No valid light curve for %s: %s", tic_id, last_error)
            return None
        
        # Run dual detector
        report = run_dual_detector(
            data=data,
            period_min_d=period_min,
            period_max_d=period_max,
            period_step_d=period_step,
            contamination=contamination,
            z_window=z_window,
            bls_top_n=bls_top_n,
        )
        
        if report.bls_best is None:
            logger.warning("No BLS detection for %s", tic_id)
            # Still create a report with null values
            return VettingReport(
                tic_id=tic_id,
                sector=str(data.metadata.get("sector", "unknown")),
                disposition="FALSE_POSITIVE",
                confidence_score=0.0,
                vetting_flags=["NO_BLS_DETECTION"],
                metrics={
                    "period_d": 0.0,
                    "t0_bjd": 0.0,
                    "duration_d": 0.0,
                    "depth_ppm": 0.0,
                    "snr": 0.0,
                },
                check_details=[],
            )
        
        # Run vetting on best BLS candidate
        bls = report.bls_best
        sector_str = str(data.metadata.get("sector", "unknown"))
        
        vetting_report = vet_candidate(
            tic_id=tic_id,
            sector=sector_str,
            time=data.time,
            flux=data.flux,
            period=bls.period_d,
            t0=bls.t0_bjd,
            duration=bls.duration_d,
            depth_ppm=bls.depth_ppm,
            snr=bls.snr,
            stellar_radius_rsun=None,
        )
        
        # Generate diagnostic plot
        if diagnostic_dir:
            generate_diagnostic_plot(data, report, vetting_report, diagnostic_dir / f"{tic_id}_diagnostic.png")
        
        # Rate limiting
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        
        return vetting_report
        
    except Exception as e:
        logger.exception("Error processing %s: %s", tic_id, e)
        return None


def generate_diagnostic_plot(
    data: LightCurveData,
    det_report: DetectionReport,
    vet_report: VettingReport,
    output_path: Path,
):
    """Generate a 4-panel diagnostic plot including vetting results."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False, constrained_layout=True)
    
    t = data.time
    f_raw = data.raw_flux
    f_det = data.flux
    z = det_report.zscore
    flags = det_report.iforest_flags
    
    # Panel 1: Raw Flux & Trend
    ax1 = axes[0]
    ax1.scatter(t, f_raw, s=3, c="#4c72b0", alpha=0.6, label="Cleaned Flux")
    trend = f_raw / f_det
    ax1.plot(t, trend, c="#c44e52", lw=1.5, label="Savitzky-Golay Trend")
    ax1.set_ylabel("Normalized Flux", fontsize=11, fontweight="bold")
    title = f"Anomaly9 Phase 2: {data.target} ({data.mission}) | "
    title += f"Disposition: {vet_report.disposition} | Confidence: {vet_report.confidence_score:.2f}"
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, ls="--", alpha=0.5)
    
    # Panel 2: Detrended Flux, BLS Model, Anomalies
    ax2 = axes[1]
    ax2.scatter(t, f_det, s=3, c="#555555", alpha=0.5, label="Detrended Flux")
    
    # Highlight anomaly events
    if det_report.events:
        for ev in det_report.events:
            mask = (t >= ev.t_start) & (t <= ev.t_end)
            if mask.any():
                ax2.scatter(t[mask], f_det[mask], s=12, c="#d95f02", zorder=5, 
                           label=f"Anomaly ({ev.kind})" if ev == det_report.events[0] else "")
    
    # BLS best-fit model
    if det_report.bls_best is not None:
        b = det_report.bls_best
        phase = ((t - data.time[0]) % b.period_d) / b.period_d
        s_idx = np.argsort(phase)
        model_flux = np.ones_like(t)
        in_transit = (phase % 1.0) < (b.duration_d / b.period_d)
        model_flux[in_transit] += (b.depth_ppm / 1e6)
        ax2.scatter(t[s_idx], model_flux[s_idx], s=1, c="#1b9e77", zorder=4, 
                   label=f"BLS Best P={b.period_d:.4f}d")
    
    ax2.set_ylabel("Detrended Flux", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, ls="--", alpha=0.5)
    
    # Panel 3: BLS Periodogram & Z-score
    ax3 = axes[2]
    if det_report.bls_best is not None and det_report.bls_best.periods is not None:
        p_arr = det_report.bls_best.periods
        pow_arr = det_report.bls_best.powers
        ax3_twin = ax3.twinx()
        ax3_twin.plot(p_arr, pow_arr, c="#7570b3", lw=1.0, label="BLS Power")
        ax3_twin.set_ylabel("BLS Power (SNR)", color="#7570b3", fontsize=10, fontweight="bold")
        ax3_twin.tick_params(axis="y", labelcolor="#7570b3")
        ax3_twin.set_ylim(bottom=0)
    
    ax3.plot(t, z, c="#333333", lw=0.8, alpha=0.7, label="Robust Rolling Z-score")
    if flags.any():
        ax3.scatter(t[flags], z[flags], s=10, c="#e7298a", zorder=5, label="Isolation Forest Flag")
    ax3.axhline(-4.0, c="#d95f02", ls=":", lw=1.2, label="Z = -4 Threshold")
    ax3.set_ylabel("Robust Z-Score", fontsize=11, fontweight="bold")
    ax3.set_xlabel(f"Time [BJD - {data.time[0]:.2f}]", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax3.grid(True, ls="--", alpha=0.5)
    
    # Panel 4: Vetting Check Results
    ax4 = axes[3]
    if vet_report.check_details:
        check_names = [c["name"] for c in vet_report.check_details]
        check_passed = [c["passed"] for c in vet_report.check_details]
        check_values = [c["value"] for c in vet_report.check_details]
        check_thresholds = [c["threshold"] for c in vet_report.check_details]
        
        y_pos = np.arange(len(check_names))
        colors = ["#1b9e77" if p else "#e7298a" for p in check_passed]
        bars = ax4.barh(y_pos, check_values, color=colors, alpha=0.7, edgecolor="black", height=0.6)
        
        # Add threshold markers
        for i, (val, thresh) in enumerate(zip(check_values, check_thresholds)):
            if thresh > 0:
                ax4.axvline(thresh, ymin=(i-0.3)/len(check_names), ymax=(i+0.3)/len(check_names), 
                           color="red", ls="--", lw=1.5, alpha=0.8)
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([n.replace("_", " ").title() for n in check_names], fontsize=10)
        ax4.set_xlabel("Metric Value", fontsize=11, fontweight="bold")
        ax4.set_title("Vetting Check Results (green=pass, red=fail, dashed=threshold)", fontsize=11, fontweight="bold")
        ax4.grid(True, ls="--", alpha=0.5, axis="x")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#1b9e77", edgecolor="black", label="Pass"),
            Patch(facecolor="#e7298a", edgecolor="black", label="Fail"),
            plt.Line2D([0], [0], color="red", ls="--", lw=1.5, label="Threshold"),
        ]
        ax4.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diagnostic plot to %s", output_path)


def write_outputs(reports: List[VettingReport], output_dir: Path):
    """Write ranked results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by confidence descending
    sorted_reports = sorted(reports, key=lambda r: r.confidence_score, reverse=True)
    
    # JSON output
    json_path = output_dir / "candidates_ranked.json"
    json_data = [r.to_dict() for r in sorted_reports]
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info("Wrote JSON results to %s", json_path)
    
    # CSV output
    csv_path = output_dir / "candidates_ranked.csv"
    if sorted_reports:
        # Flatten for CSV
        fieldnames = [
            "rank", "tic_id", "sector", "disposition", "confidence_score",
            "period_d", "t0_bjd", "duration_d", "depth_ppm", "snr",
            "vetting_flags",
        ]
        # Add all possible metric keys
        all_metric_keys = set()
        for r in sorted_reports:
            all_metric_keys.update(r.metrics.keys())
        for k in sorted(all_metric_keys):
            if k not in fieldnames:
                fieldnames.append(k)
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, r in enumerate(sorted_reports, 1):
                row = {
                    "rank": rank,
                    "tic_id": r.tic_id,
                    "sector": r.sector,
                    "disposition": r.disposition,
                    "confidence_score": f"{r.confidence_score:.4f}",
                    "vetting_flags": ";".join(r.vetting_flags),
                }
                row.update({k: f"{v:.6f}" if isinstance(v, float) else str(v) 
                           for k, v in r.metrics.items()})
                writer.writerow(row)
        logger.info("Wrote CSV results to %s", csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly9 Phase 2: Multi-Target Batch Pipeline with Astrophysical Vetting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Target selection
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--tic",
        type=str,
        help="Single TIC ID to process (e.g., '25155310' or 'TIC 25155310')",
    )
    target_group.add_argument(
        "--catalog",
        type=str,
        help="Path to catalog file (newline-separated TIC IDs or CSV with 'tic_id' column)",
    )
    
    # Mission & sector
    parser.add_argument(
        "--mission",
        type=str,
        default="TESS",
        choices=["TESS", "Kepler"],
        help="Mission name",
    )
    parser.add_argument(
        "--sectors",
        type=str,
        default=None,
        help="Comma-separated sector list (e.g., '1,2,3'). Default: all available.",
    )
    
    # Detection parameters
    parser.add_argument(
        "--period-min",
        type=float,
        default=0.5,
        help="Minimum trial period [days]",
    )
    parser.add_argument(
        "--period-max",
        type=float,
        default=10.0,
        help="Maximum trial period [days]",
    )
    parser.add_argument(
        "--period-step",
        type=float,
        default=0.002,
        help="Period step [days]",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Isolation Forest contamination rate",
    )
    parser.add_argument(
        "--z-window",
        type=int,
        default=25,
        help="Rolling Z-score window size",
    )
    parser.add_argument(
        "--bls-top-n",
        type=int,
        default=5,
        help="Number of top BLS peaks to keep",
    )
    
    # Batch processing
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (1 = sequential)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Delay between MAST queries [seconds]",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=str,
        default="diagnostics",
        help="Directory for diagnostic plots",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Cache directory for FITS files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Parse sectors
    sectors = None
    if args.sectors:
        sectors = [int(s.strip()) for s in args.sectors.split(",")]
    
    # Parse target list
    tic_ids = []
    if args.tic:
        tic_id = args.tic.replace("TIC", "").replace("tic", "").strip()
        tic_ids = [tic_id]
    elif args.catalog:
        tic_ids = parse_catalog(Path(args.catalog))
    
    if not tic_ids:
        logger.error("No valid TIC IDs to process")
        sys.exit(1)
    
    logger.info("Processing %d target(s) with %d worker(s)", len(tic_ids), args.workers)
    
    # Prepare directories
    output_dir = Path(args.output_dir)
    diagnostics_dir = Path(args.diagnostics_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Process targets
    reports: List[VettingReport] = []
    
    if args.workers == 1:
        # Sequential processing
        for tic_id in tic_ids:
            report = process_single_target(
                tic_id=tic_id,
                mission=args.mission,
                sectors=sectors,
                period_min=args.period_min,
                period_max=args.period_max,
                period_step=args.period_step,
                contamination=args.contamination,
                z_window=args.z_window,
                bls_top_n=args.bls_top_n,
                output_dir=output_dir,
                diagnostic_dir=diagnostics_dir,
                cache_dir=cache_dir,
                rate_limit_delay=args.rate_limit,
            )
            if report:
                reports.append(report)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_single_target,
                    tic_id=tic_id,
                    mission=args.mission,
                    sectors=sectors,
                    period_min=args.period_min,
                    period_max=args.period_max,
                    period_step=args.period_step,
                    contamination=args.contamination,
                    z_window=args.z_window,
                    bls_top_n=args.bls_top_n,
                    output_dir=output_dir,
                    diagnostic_dir=diagnostics_dir,
                    cache_dir=cache_dir,
                    rate_limit_delay=args.rate_limit,
                ): tic_id
                for tic_id in tic_ids
            }
            
            for future in as_completed(futures):
                tic_id = futures[future]
                try:
                    report = future.result()
                    if report:
                        reports.append(report)
                except Exception as e:
                    logger.exception("Worker error for %s: %s", tic_id, e)
    
    # Write outputs
    if reports:
        write_outputs(reports, output_dir)
        
        # Print summary
        print("\n" + "=" * 80)
        print(f" ANOMALY9 BATCH SUMMARY: {len(reports)} targets processed")
        print("=" * 80)
        print(f" {'Rank':<5} {'TIC ID':<15} {'Sector':<8} {'Disposition':<15} {'Confidence':<10} {'Flags'}")
        print("-" * 80)
        for i, r in enumerate(sorted(reports, key=lambda x: x.confidence_score, reverse=True), 1):
            flags_str = ", ".join(r.vetting_flags) if r.vetting_flags else "none"
            print(f" {i:<5} {r.tic_id:<15} {r.sector:<8} {r.disposition:<15} {r.confidence_score:<10.3f} {flags_str}")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}/candidates_ranked.json")
        print(f"                 {output_dir}/candidates_ranked.csv")
        if diagnostics_dir.exists():
            print(f"Diagnostics saved to: {diagnostics_dir}/")
    else:
        logger.warning("No successful results to report")
        sys.exit(1)


if __name__ == "__main__":
    main()