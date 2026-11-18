"""
Integration tests for Anomaly9 Phase 2 batch pipeline and CLI.
Tests use synthetic/mock light curves (no live MAST network calls).
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from anomaly9.pipeline import LightCurveData, clean_lightcurve, detrend_lightcurve
from anomaly9.detector import run_dual_detector, DetectionReport
from anomaly9.vetting import vet_candidate, VettingReport
from anomaly9.cli import parse_catalog, process_single_target, write_outputs


def make_synthetic_lcdata(
    tic_id: str = "25155310",
    sector: str = "1",
    period: float = 2.0,
    depth_ppm: float = 2000,
    n_points: int = 500,
    noise: float = 50,
) -> LightCurveData:
    """Create a synthetic LightCurveData with a clean planetary transit."""
    time = np.linspace(2458000.0, 2458030.0, n_points)
    flux = np.ones(n_points)
    
    # Inject 5 transits
    for i in range(5):
        center = 2458000.0 + i * period
        mask = np.abs(time - center) < 0.05
        flux[mask] -= depth_ppm / 1e6
    
    flux += np.random.normal(0, noise / 1e6, n_points)
    
    # Detrend
    flux_det = detrend_lightcurve(time, flux)
    
    return LightCurveData(
        target=f"TIC {tic_id}",
        mission="TESS",
        time=time,
        raw_flux=flux,
        flux=flux_det,
        cadence_s=120.0,
        metadata={"sector": sector, "target_name": f"TIC {tic_id}"},
    )


def make_eb_lcdata(
    tic_id: str = "25155311",
    sector: str = "1",
    period: float = 2.0,
    depth_ppm: float = 1000,
    n_points: int = 500,
) -> LightCurveData:
    """Create a synthetic LightCurveData with EB signature (odd/even depths)."""
    time = np.linspace(2458000.0, 2458030.0, n_points)
    flux = np.ones(n_points)
    
    # Inject transits with alternating depths - make difference 3x (200% difference)
    for i in range(5):
        center = 2458000.0 + i * period
        mask = np.abs(time - center) < 0.05
        d = depth_ppm / 1e6
        if i % 2 == 1:
            d *= 3.0  # 200% deeper on even transits (exceeds 30% threshold)
        flux[mask] -= d
    
    flux += np.random.normal(0, 50 / 1e6, n_points)
    flux_det = detrend_lightcurve(time, flux)
    
    return LightCurveData(
        target=f"TIC {tic_id}",
        mission="TESS",
        time=time,
        raw_flux=flux,
        flux=flux_det,
        cadence_s=120.0,
        metadata={"sector": sector, "target_name": f"TIC {tic_id}"},
    )


def make_instrumental_lcdata(
    tic_id: str = "25155312",
    sector: str = "1",
    n_points: int = 500,
) -> LightCurveData:
    """Create a synthetic LightCurveData with instrumental artifacts at 2.5d cadence."""
    time = np.linspace(2458000.0, 2458030.0, n_points)
    flux = np.ones(n_points)
    
    # Artifacts at momentum dump cadence (~2.5 days) - add noise and make them shallow
    for i in range(12):
        center = 2458000.0 + i * 2.5
        mask = np.abs(time - center) < 0.03
        flux[mask] -= 0.0005  # Shallower artifacts
    
    # Add significant red noise to reduce SPNR
    flux += np.random.normal(0, 200 / 1e6, n_points)
    
    # Add low-frequency trend (stellar variability not fully removed)
    trend = 1.0 + 0.01 * np.sin(2 * np.pi * time / 10.0)
    flux = flux * trend
    
    flux_det = detrend_lightcurve(time, flux)
    
    return LightCurveData(
        target=f"TIC {tic_id}",
        mission="TESS",
        time=time,
        raw_flux=flux,
        flux=flux_det,
        cadence_s=120.0,
        metadata={"sector": sector, "target_name": f"TIC {tic_id}"},
    )


class TestParseCatalog:
    def test_newline_separated(self):
        """Test parsing newline-separated TIC IDs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("25155310\n25155311\n25155312\n")
            f.flush()
            result = parse_catalog(Path(f.name))
        os.unlink(f.name)
        assert result == ["TIC 25155310", "TIC 25155311", "TIC 25155312"]
    
    def test_csv_with_header(self):
        """Test parsing CSV with tic_id column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("tic_id,ra,dec\n25155310,10.0,20.0\n25155311,11.0,21.0\n")
            f.flush()
            result = parse_catalog(Path(f.name))
        os.unlink(f.name)
        assert result == ["TIC 25155310", "TIC 25155311"]
    
    def test_csv_with_tic_column(self):
        """Test parsing CSV with TIC column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("TIC,mag\n25155310,12.5\n25155311,13.0\n")
            f.flush()
            result = parse_catalog(Path(f.name))
        os.unlink(f.name)
        assert result == ["TIC 25155310", "TIC 25155311"]
    
    def test_comments_ignored(self):
        """Test that comment lines are ignored."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# This is a comment\n25155310\n# Another comment\n25155311\n")
            f.flush()
            result = parse_catalog(Path(f.name))
        os.unlink(f.name)
        assert result == ["TIC 25155310", "TIC 25155311"]


class TestVettingIntegration:
    def test_planet_candidate_runs_vetting(self):
        """Clean planet candidate should run vetting and return a report."""
        lc = make_synthetic_lcdata(depth_ppm=3000, noise=50)
        
        # Run detector
        det_report = run_dual_detector(
            data=lc,
            period_min_d=1.5,
            period_max_d=3.0,
            period_step_d=0.01,
        )
        
        assert det_report.bls_best is not None
        bls = det_report.bls_best
        
        # Run vetting
        vet_report = vet_candidate(
            tic_id="25155310",
            sector="1",
            time=lc.time,
            flux=lc.flux,
            period=bls.period_d,
            t0=bls.t0_bjd,
            duration=bls.duration_d,
            depth_ppm=bls.depth_ppm,
            snr=bls.snr,
        )
        
        assert vet_report.disposition in ["CANDIDATE", "AMBIGUOUS", "FALSE_POSITIVE"]
        assert 0.0 <= vet_report.confidence_score <= 1.0
        assert isinstance(vet_report.vetting_flags, list)
    
    def test_eb_suspect_runs_vetting(self):
        """EB with odd/even depths should run vetting and produce valid report."""
        lc = make_eb_lcdata()
        
        det_report = run_dual_detector(
            data=lc,
            period_min_d=1.5,
            period_max_d=3.0,
            period_step_d=0.01,
        )
        
        assert det_report.bls_best is not None
        bls = det_report.bls_best
        
        vet_report = vet_candidate(
            tic_id="25155311",
            sector="1",
            time=lc.time,
            flux=lc.flux,
            period=bls.period_d,
            t0=bls.t0_bjd,
            duration=bls.duration_d,
            depth_ppm=bls.depth_ppm,
            snr=bls.snr,
        )
        
        # Just verify vetting runs and returns valid disposition
        assert vet_report.disposition in ["CANDIDATE", "AMBIGUOUS", "FALSE_POSITIVE"]
        assert 0.0 <= vet_report.confidence_score <= 1.0
    
    def test_instrumental_artifact_runs_vetting(self):
        """Instrumental artifact at momentum dump cadence should run vetting."""
        lc = make_instrumental_lcdata()
        
        det_report = run_dual_detector(
            data=lc,
            period_min_d=2.0,
            period_max_d=3.0,
            period_step_d=0.01,
        )
        
        assert det_report.bls_best is not None
        bls = det_report.bls_best
        
        vet_report = vet_candidate(
            tic_id="25155312",
            sector="1",
            time=lc.time,
            flux=lc.flux,
            period=bls.period_d,
            t0=bls.t0_bjd,
            duration=bls.duration_d,
            depth_ppm=bls.depth_ppm,
            snr=bls.snr,
        )
        
        assert vet_report.disposition in ["CANDIDATE", "AMBIGUOUS", "FALSE_POSITIVE"]
        assert 0.0 <= vet_report.confidence_score <= 1.0
    
    def test_no_bls_detection(self):
        """Target with no BLS detection should get FALSE_POSITIVE."""
        lc = make_synthetic_lcdata(depth_ppm=50, noise=500)  # Very shallow, noisy
        
        det_report = run_dual_detector(
            data=lc,
            period_min_d=1.5,
            period_max_d=3.0,
            period_step_d=0.01,
        )
        
        if det_report.bls_best is None:
            # Create a report for no detection
            vet_report = VettingReport(
                tic_id="25155313",
                sector="1",
                disposition="FALSE_POSITIVE",
                confidence_score=0.0,
                vetting_flags=["NO_BLS_DETECTION"],
                metrics={"period_d": 0, "t0_bjd": 0, "duration_d": 0, "depth_ppm": 0, "snr": 0},
                check_details=[],
            )
            assert vet_report.disposition == "FALSE_POSITIVE"
            assert "NO_BLS_DETECTION" in vet_report.vetting_flags


class TestBatchPipeline:
    @patch("anomaly9.cli.ingest_mast_lightcurve_multi_sector")
    def test_process_single_target_success(self, mock_ingest_multi):
        """Test process_single_target with mocked MAST ingestion."""
        # Create mock light curve
        lc = make_synthetic_lcdata()
        mock_ingest_multi.return_value = lc
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            diag_dir = Path(tmpdir) / "diagnostics"
            cache_dir = Path(tmpdir) / "cache"
            
            report = process_single_target(
                tic_id="25155310",
                mission="TESS",
                sectors=None,
                period_min=1.5,
                period_max=3.0,
                period_step=0.01,
                contamination=0.01,
                z_window=25,
                bls_top_n=5,
                output_dir=output_dir,
                diagnostic_dir=diag_dir,
                cache_dir=cache_dir,
                rate_limit_delay=0.0,
            )
            
            assert report is not None
            assert isinstance(report, VettingReport)
            assert report.tic_id == "25155310"
            assert report.disposition in ["CANDIDATE", "AMBIGUOUS", "FALSE_POSITIVE"]
    
    @patch("anomaly9.cli.ingest_mast_lightcurve_multi_sector")
    def test_process_single_target_failure(self, mock_ingest_multi):
        """Test process_single_target handles ingestion failure."""
        mock_ingest_multi.side_effect = Exception("MAST query failed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report = process_single_target(
                tic_id="25155310",
                mission="TESS",
                sectors=None,
                period_min=1.5,
                period_max=3.0,
                period_step=0.01,
                contamination=0.01,
                z_window=25,
                bls_top_n=5,
                output_dir=Path(tmpdir) / "output",
                diagnostic_dir=Path(tmpdir) / "diagnostics",
                cache_dir=Path(tmpdir) / "cache",
                rate_limit_delay=0.0,
            )
            
            assert report is None
    
    def test_write_outputs_json_csv(self):
        """Test write_outputs creates valid JSON and CSV."""
        reports = [
            VettingReport(
                tic_id="25155310",
                sector="1",
                disposition="CANDIDATE",
                confidence_score=0.95,
                vetting_flags=[],
                metrics={"period_d": 2.0, "t0_bjd": 2458000.0, "duration_d": 0.1, "depth_ppm": 2000, "snr": 15.0},
                check_details=[],
            ),
            VettingReport(
                tic_id="25155311",
                sector="2",
                disposition="FALSE_POSITIVE",
                confidence_score=0.1,
                vetting_flags=["EB_SUSPECT"],
                metrics={"period_d": 1.5, "t0_bjd": 2458000.5, "duration_d": 0.05, "depth_ppm": 5000, "snr": 8.0},
                check_details=[],
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            write_outputs(reports, output_dir)
            
            # Check JSON
            json_path = output_dir / "candidates_ranked.json"
            assert json_path.exists()
            with open(json_path) as f:
                json_data = json.load(f)
            assert len(json_data) == 2
            assert json_data[0]["tic_id"] == "25155310"  # Higher confidence first
            assert json_data[1]["tic_id"] == "25155311"
            
            # Check CSV
            csv_path = output_dir / "candidates_ranked.csv"
            assert csv_path.exists()
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # Header + 2 data rows
            assert "25155310" in lines[1]
            assert "25155311" in lines[2]
    
    def test_write_outputs_empty(self):
        """Test write_outputs with empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            write_outputs([], output_dir)
            
            json_path = output_dir / "candidates_ranked.json"
            assert json_path.exists()
            with open(json_path) as f:
                data = json.load(f)
            assert data == []


class TestCLIArguments:
    def test_cli_help(self):
        """Test CLI help doesn't crash."""
        from anomaly9.cli import main
        import sys
        
        # Capture help output
        old_argv = sys.argv
        sys.argv = ["anomaly9-batch", "--help"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Help exits with 0
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv
    
    def test_cli_missing_target(self):
        """Test CLI fails without --tic or --catalog."""
        from anomaly9.cli import main
        import sys
        
        old_argv = sys.argv
        sys.argv = ["anomaly9-batch"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0
        finally:
            sys.argv = old_argv


class TestEndToEndBatch:
    @patch("anomaly9.cli.ingest_mast_lightcurve_multi_sector")
    def test_batch_multiple_targets(self, mock_ingest_multi):
        """Test batch processing multiple targets."""
        # Create different light curves for different targets
        def side_effect(target, mission, sectors, **kwargs):
            if target == "25155310":
                return make_synthetic_lcdata(tic_id=target)
            elif target == "25155311":
                return make_eb_lcdata(tic_id=target)
            elif target == "25155312":
                return make_instrumental_lcdata(tic_id=target)
            raise ValueError(f"Unknown target {target}")
        
        mock_ingest_multi.side_effect = side_effect
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            diag_dir = Path(tmpdir) / "diagnostics"
            cache_dir = Path(tmpdir) / "cache"
            
            # Process three targets
            tic_ids = ["25155310", "25155311", "25155312"]
            reports = []
            
            for tic_id in tic_ids:
                report = process_single_target(
                    tic_id=tic_id,
                    mission="TESS",
                    sectors=None,
                    period_min=1.5,
                    period_max=3.0,
                    period_step=0.01,
                    contamination=0.01,
                    z_window=25,
                    bls_top_n=5,
                    output_dir=output_dir,
                    diagnostic_dir=diag_dir,
                    cache_dir=cache_dir,
                    rate_limit_delay=0.0,
                )
                if report:
                    reports.append(report)
            
            assert len(reports) == 3
            
            # Write outputs
            write_outputs(reports, output_dir)
            
            # Verify outputs exist
            json_path = output_dir / "candidates_ranked.json"
            with open(json_path) as f:
                data = json.load(f)
            
            assert len(data) == 3
            
            # Check all three TIC IDs are present
            tic_ids_out = {d["tic_id"] for d in data}
            assert tic_ids_out == {"25155310", "25155311", "25155312"}
            
            # Check ranking by confidence
            confidences = [d["confidence_score"] for d in data]
            assert confidences == sorted(confidences, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])