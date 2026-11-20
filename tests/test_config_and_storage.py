"""
Tests for Anomaly9 Configuration System and Storage
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from anomaly9.config import (
    Anomaly9Config,
    CatalogConfig,
    DatabaseConfig,
    DetectorConfig,
    MastConfig,
    OutputConfig,
    VettingConfig,
    generate_default_toml,
    load_config,
)
from anomaly9.pipeline import LightCurveData
from anomaly9.storage import CandidateDB
from anomaly9.vetting import VettingReport, VettingCheckResult


class TestConfigDataclasses:
    """Tests for configuration dataclasses."""
    
    def test_default_configs(self):
        """Test that default configs have expected values."""
        config = Anomaly9Config()
        
        assert config.mast.rate_limit == 1.0
        assert config.mast.max_sectors == 6
        assert config.detector.period_min == 0.5
        assert config.detector.period_max == 10.0
        assert config.detector.period_step == 0.002
        assert config.vetting.snr_threshold == 7.0
        assert config.catalog.enable_gaia is True
        assert config.catalog.enable_tic is True
        assert config.catalog.enable_exofop is True
        assert config.database.url == "sqlite:///candidates.db"
        assert config.database.wal_mode is True
    
    def test_config_to_dict(self):
        """Test serialization to dictionary."""
        config = Anomaly9Config()
        d = config.to_dict()
        
        assert "mast" in d
        assert "detector" in d
        assert "vetting" in d
        assert "catalog" in d
        assert "output" in d
        assert "database" in d
    
    def test_config_from_dict(self):
        """Test deserialization from dictionary."""
        config = Anomaly9Config()
        d = config.to_dict()
        config2 = Anomaly9Config.from_dict(d)
        
        assert config.mast.rate_limit == config2.mast.rate_limit
        assert config.detector.period_min == config2.detector.period_min
    
    def test_config_hash(self):
        """Test that config hash is stable."""
        config1 = Anomaly9Config()
        config2 = Anomaly9Config()
        
        assert config1.config_hash == config2.config_hash
        
        # Different config should have different hash
        config3 = Anomaly9Config()
        config3.mast.rate_limit = 2.0
        assert config3.config_hash != config1.config_hash
    
    def test_config_to_pipeline_kwargs(self):
        """Test extracting pipeline kwargs from config."""
        from anomaly9.config import config_to_pipeline_kwargs
        
        config = Anomaly9Config()
        kwargs = config_to_pipeline_kwargs(config)
        
        assert kwargs["period_min"] == 0.5
        assert kwargs["period_max"] == 10.0
        assert kwargs["enable_gaia"] is True
        assert kwargs["enable_tic"] is True
        assert kwargs["enable_exofop"] is True


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_config_defaults(self):
        """Test loading config with no file (defaults)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp dir to avoid finding any existing config
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(tmpdir)
                config = load_config()
                assert config.mast.rate_limit == 1.0
                assert config.detector.period_min == 0.5
            finally:
                os.chdir(old_cwd)
    
    def test_load_config_from_toml(self, tmp_path):
        """Test loading config from TOML file."""
        toml_path = tmp_path / "anomaly9.toml"
        toml_content = """
[mast]
rate_limit = 2.0
max_sectors = 3

[detector]
period_min = 1.0
period_max = 5.0
"""
        toml_path.write_text(toml_content)
        
        config = load_config(config_path=toml_path)
        
        assert config.mast.rate_limit == 2.0
        assert config.mast.max_sectors == 3
        assert config.detector.period_min == 1.0
        assert config.detector.period_max == 5.0
        # Unspecified should keep defaults
        assert config.detector.period_step == 0.002
    
    def test_load_config_cli_overrides(self, tmp_path):
        """Test CLI overrides take precedence over TOML."""
        toml_path = tmp_path / "anomaly9.toml"
        toml_content = """
[mast]
rate_limit = 2.0

[detector]
period_min = 1.0
"""
        toml_path.write_text(toml_content)
        
        config = load_config(
            config_path=toml_path,
            cli_overrides={"mast.rate_limit": 3.0, "detector.period_max": 20.0}
        )
        
        # CLI override wins
        assert config.mast.rate_limit == 3.0
        # TOML value used when no CLI override
        assert config.detector.period_min == 1.0
        # Default used when neither specified
        assert config.detector.period_max == 20.0
    
    def test_load_config_none_file(self):
        """Test loading with explicit None config path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(tmpdir)
                config = load_config(config_path=None)
                assert config.mast.rate_limit == 1.0
            finally:
                os.chdir(old_cwd)
    
    def test_load_config_missing_file(self):
        """Test error when config file not found."""
        with pytest.raises(FileNotFoundError):
            load_config(config_path="/nonexistent/anomaly9.toml")


class TestGenerateDefaultToml:
    """Tests for generate_default_toml function."""
    
    def test_generate_default_toml(self, tmp_path):
        """Test generating default TOML config file."""
        output_path = tmp_path / "test_config.toml"
        generate_default_toml(output_path)
        
        assert output_path.exists()
        content = output_path.read_text()
        
        # Check key sections present
        assert "[mast]" in content
        assert "[detector]" in content
        assert "[vetting]" in content
        assert "[catalog]" in content
        assert "[output]" in content
        assert "[database]" in content
        
        # Check some default values
        assert "rate_limit = 1.0" in content
        assert "period_min = 0.5" in content
        assert "snr_threshold = 7.0" in content


class TestCandidateDB:
    """Tests for CandidateDB storage."""
    
    def _make_report(self, tic_id="TIC 12345678", disposition="CANDIDATE", score=0.9):
        """Create a mock VettingReport."""
        checks = [
            VettingCheckResult("snr_check", True, 10.0, 7.0, "OK"),
            VettingCheckResult("odd_even_check", True, 0.1, 0.3, "OK"),
        ]
        return VettingReport(
            tic_id=tic_id,
            sector="1",
            disposition=disposition,
            confidence_score=score,
            vetting_flags=[],
            metrics={
                "period_d": 3.5,
                "t0_bjd": 2459000.5,
                "duration_d": 0.1,
                "depth_ppm": 500,
                "snr": 10.0,
                "isolation_score": 0.8,
            },
            check_details=checks,
        )
    
    def _make_lc_data(self, tic_id="TIC 12345678"):
        """Create a mock LightCurveData with catalog metadata."""
        lc_data = LightCurveData(
            target=tic_id,
            mission="TESS",
            time=np.arange(1000, dtype=float),
            raw_flux=np.ones(1000),
            flux=np.ones(1000),
        )
        # Add stellar metadata
        from anomaly9.catalog import StellarMetadata
        lc_data.metadata["stellar_metadata"] = StellarMetadata(
            tic_id=tic_id,
            radius_solar=1.0,
            mass_solar=1.0,
            teff_k=5778,
        )
        return lc_data
    
    def test_db_init(self, tmp_path):
        """Test database initialization."""
        db_path = tmp_path / "test.db"
        db = CandidateDB(url=f"sqlite:///{db_path}")
        
        assert db_path.exists()
        db.close()
    
    def test_upsert_and_get_candidate(self, tmp_path):
        """Test upserting and retrieving a candidate."""
        db_path = tmp_path / "test.db"
        db = CandidateDB(url=f"sqlite:///{db_path}")
        
        report = self._make_report("TIC 12345678")
        lc_data = self._make_lc_data("TIC 12345678")
        
        db.upsert_candidate(report, lc_data, "run_001")
        
        candidate = db.get_by_tic("TIC 12345678")
        
        assert candidate is not None
        assert candidate["tic_id"] == "TIC 12345678"
        assert candidate["disposition"] == "CANDIDATE"
        assert candidate["vetting_score"] == 0.9
        assert candidate["period_days"] == 3.5
        assert candidate["stellar_radius"] == 1.0
        assert candidate["stellar_density"] == pytest.approx(1.408, rel=0.01)
        assert candidate["checks"] is not None
        assert len(candidate["checks"]) == 2
        
        db.close()
    
    def test_upsert_overwrites(self, tmp_path):
        """Test that upsert overwrites existing candidate."""
        db_path = tmp_path / "test.db"
        db = CandidateDB(url=f"sqlite:///{db_path}")
        
        report1 = self._make_report("TIC 12345678", "CANDIDATE", 0.9)
        report2 = self._make_report("TIC 12345678", "FALSE_POSITIVE", 0.1)
        lc_data = self._make_lc_data("TIC 12345678")
        
        db.upsert_candidate(report1, lc_data, "run_001")
        db.upsert_candidate(report2, lc_data, "run_002")
        
        candidate = db.get_by_tic("TIC 12345678")
        
        assert candidate["disposition"] == "FALSE_POSITIVE"
        assert candidate["vetting_score"] == 0.1
        assert candidate["run_id"] == "run_002"
        
        db.close()
    
    def test_query_candidates(self, tmp_path):
        """Test querying candidates with filters."""
        db_path = tmp_path / "test.db"
        db = CandidateDB(url=f"sqlite:///{db_path}")
        
        # Insert multiple candidates
        for i, (tid, disp, score) in enumerate([
            ("TIC 1", "CANDIDATE", 0.9),
            ("TIC 2", "AMBIGUOUS", 0.7),
            ("TIC 3", "FALSE_POSITIVE", 0.3),
            ("TIC 4", "CANDIDATE", 0.8),
        ]):
            report = self._make_report(tid, disp, score)
            lc_data = self._make_lc_data(tid)
            db.upsert_candidate(report, lc_data, f"run_{i}")
        
        # Query by disposition
        candidates = db.query_candidates(disposition="CANDIDATE")
        assert len(candidates) == 2
        for c in candidates:
            assert c["disposition"] == "CANDIDATE"
        
        # Query by min score
        candidates = db.query_candidates(min_score=0.75)
        assert len(candidates) == 2  # 0.9 and 0.8
        for c in candidates:
            assert c["vetting_score"] >= 0.75
        
        # Query with limit
        candidates = db.query_candidates(limit=2)
        assert len(candidates) == 2
        
        db.close()
    
    def test_mark_reviewed(self, tmp_path):
        """Test marking candidate as reviewed."""
        db_path = tmp_path / "test.db"
        db = CandidateDB(url=f"sqlite:///{db_path}")
        
        report = self._make_report("TIC 12345678")
        lc_data = self._make_lc_data("TIC 12345678")
        db.upsert_candidate(report, lc_data, "run_001")
        
        # Initially not reviewed
        candidate = db.get_by_tic("TIC 12345678")
        assert candidate["reviewed"] == 0
        
        # Mark reviewed
        result = db.mark_reviewed("TIC 12345678", "Looks good")
        assert result is True
        
        candidate = db.get_by_tic("TIC 12345678")
        assert candidate["reviewed"] == 1
        assert candidate["reviewer_notes"] == "Looks good"
        
        db.close()
    
    def test_mark_reviewed_nonexistent(self, tmp_path):
        """Test marking non-existent candidate returns False."""
        db_path = tmp_path / "test.db"
        db = CandidateDB(url=f"sqlite:///{db_path}")
        
        result = db.mark_reviewed("TIC 99999999", "Notes")
        assert result is False
        
        db.close()
    
    def test_export_to_csv_json(self, tmp_path):
        """Test exporting to CSV and JSON."""
        db_path = tmp_path / "test.db"
        db = CandidateDB(url=f"sqlite:///{db_path}")
        output_dir = tmp_path / "output"
        
        # Insert candidates
        for i, (tid, disp, score) in enumerate([
            ("TIC 1", "CANDIDATE", 0.9),
            ("TIC 2", "FALSE_POSITIVE", 0.2),
        ]):
            report = self._make_report(tid, disp, score)
            lc_data = self._make_lc_data(tid)
            db.upsert_candidate(report, lc_data, f"run_{i}")
        
        csv_path, json_path = db.export_to_csv_json(output_dir)
        
        assert csv_path.exists()
        assert json_path.exists()
        assert csv_path.name == "candidates_ranked.csv"
        assert json_path.name == "candidates_ranked.json"
        
        # Verify CSV content
        csv_content = csv_path.read_text()
        assert "TIC 1" in csv_content
        assert "CANDIDATE" in csv_content
        
        # Verify JSON content
        json_content = json.loads(json_path.read_text())
        assert len(json_content) == 2
        assert json_content[0]["tic_id"] == "TIC 1"
        
        db.close()
    
    def test_count_candidates(self, tmp_path):
        """Test counting candidates."""
        db_path = tmp_path / "test.db"
        db = CandidateDB(url=f"sqlite:///{db_path}")
        
        assert db.count_candidates() == 0
        
        for tid in ["TIC 1", "TIC 2", "TIC 3"]:
            report = self._make_report(tid, "CANDIDATE")
            lc_data = self._make_lc_data(tid)
            db.upsert_candidate(report, lc_data, "run_1")
        
        assert db.count_candidates() == 3
        assert db.count_candidates("CANDIDATE") == 3
        assert db.count_candidates("FALSE_POSITIVE") == 0
        
        db.close()
    
    def test_in_memory_db(self):
        """Test in-memory database."""
        db = CandidateDB(url="sqlite:///:memory:")
        
        report = self._make_report("TIC 12345678")
        lc_data = self._make_lc_data("TIC 12345678")
        db.upsert_candidate(report, lc_data, "run_001")
        
        candidate = db.get_by_tic("TIC 12345678")
        assert candidate is not None
        
        db.close()


class TestIntegration:
    """Integration tests for config + storage + pipeline."""
    
    @patch("anomaly9.catalog.query_tic_v9")
    @patch("anomaly9.catalog.query_exofop_tess")
    def test_config_db_pipeline_integration(self, mock_exofop, mock_tic, tmp_path):
        """Test full integration: config -> db -> pipeline."""
        # Mock catalog queries
        mock_tic.return_value = None
        mock_exofop.return_value = None
        
        # Create config with test settings
        config = Anomaly9Config()
        config.mast.max_sectors = 1
        config.detector.period_min = 1.0
        config.detector.period_max = 5.0
        
        # Initialize database
        db_path = tmp_path / "test.db"
        db = CandidateDB(url=f"sqlite:///{db_path}")
        
        # Create test light curve data
        from anomaly9.pipeline import LightCurveData
        lc_data = LightCurveData(
            target="TIC 12345678",
            mission="TESS",
            time=np.arange(100, dtype=float),
            raw_flux=np.ones(100),
            flux=np.ones(100),
        )
        
        # Create a mock report
        from anomaly9.vetting import VettingReport, VettingCheckResult
        checks = [VettingCheckResult("test_check", True, 1.0, 1.0, "OK")]
        report = VettingReport(
            tic_id="TIC 12345678",
            sector="1",
            disposition="CANDIDATE",
            confidence_score=0.85,
            vetting_flags=[],
            metrics={
                "period_d": 3.0,
                "t0_bjd": 2459000.0,
                "duration_d": 0.1,
                "depth_ppm": 500,
                "snr": 8.0,
                "isolation_score": 0.9,
            },
            check_details=checks,
        )
        
        # Test upsert
        db.upsert_candidate(report, lc_data, "run_test")
        
        # Verify stored
        candidate = db.get_by_tic("TIC 12345678")
        assert candidate is not None
        assert candidate["disposition"] == "CANDIDATE"
        assert candidate["vetting_score"] == 0.85
        
        # Test export
        csv_path, json_path = db.export_to_csv_json(tmp_path)
        assert csv_path.exists()
        assert json_path.exists()
        
        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])