"""
Tests for Anomaly9 Catalog Cross-Match Engine
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from anomaly9.catalog import (
    CatalogCache,
    ExofopDisposition,
    StellarMetadata,
    enrich_with_catalogs,
    get_exofop_disposition,
    get_ruwe,
    get_stellar_density,
    get_stellar_radius,
    is_known_fp,
    is_known_planet,
    query_exofop_tess,
    query_gaia_dr3_by_coords,
    query_tic_v9,
    query_gaia_dr3,
    Vizier,
    Gaia,
    _cache as global_cache,
)
from anomaly9.pipeline import LightCurveData
from anomaly9.vetting.checks import duration_period_consistency


# Helper functions to create mocks
def _make_vizier_result(rows):
    """Create a mock Vizier result."""
    mock_table = MagicMock()
    mock_table.__len__ = MagicMock(return_value=len(rows))
    mock_table.colnames = list(rows[0].keys()) if rows else []
    mock_table.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return [mock_table]


def _make_gaia_result(rows):
    """Create a mock Gaia cone search result."""
    mock_result = MagicMock()
    mock_result.__len__ = MagicMock(return_value=len(rows))
    if rows:
        mock_result.colnames = list(rows[0].keys())
    else:
        mock_result.colnames = []
    mock_result.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    
    mock_job = MagicMock()
    mock_job.get_results.return_value = mock_result
    return mock_job


def _clear_global_catalog_cache():
    """Clear the global catalog cache to avoid test interference."""
    for path in global_cache.cache_dir.glob("*.pkl"):
        try:
            path.unlink()
        except OSError:
            pass


class TestStellarMetadata:
    """Tests for StellarMetadata dataclass."""
    
    def test_stellar_density_computed_from_mass_radius(self):
        """Test that stellar density is auto-computed from mass and radius."""
        meta = StellarMetadata(
            tic_id="TIC 12345678",
            mass_solar=1.0,
            radius_solar=1.0,
        )
        # Solar density = 1.408 g/cm³
        assert meta.stellar_density_gcc is not None
        assert abs(meta.stellar_density_gcc - 1.408) < 0.01
    
    def test_stellar_density_with_non_solar_values(self):
        """Test density computation with non-solar values."""
        # M=2, R=1.5 -> density = 2/1.5^3 * 1.408 = 2/3.375 * 1.408 ≈ 0.834
        meta = StellarMetadata(
            tic_id="TIC 12345678",
            mass_solar=2.0,
            radius_solar=1.5,
        )
        expected = (2.0 / (1.5 ** 3)) * 1.408
        assert meta.stellar_density_gcc is not None
        assert abs(meta.stellar_density_gcc - expected) < 0.01
    
    def test_no_density_when_missing_mass_or_radius(self):
        """Test that density is None when mass or radius is missing."""
        meta1 = StellarMetadata(tic_id="TIC 1", mass_solar=1.0)
        assert meta1.stellar_density_gcc is None
        
        meta2 = StellarMetadata(tic_id="TIC 1", radius_solar=1.0)
        assert meta2.stellar_density_gcc is None
        
        meta3 = StellarMetadata(tic_id="TIC 1", mass_solar=1.0, radius_solar=0.0)
        assert meta3.stellar_density_gcc is None


class TestExofopDisposition:
    """Tests for ExofopDisposition dataclass."""
    
    def test_known_fp_flag_explicit(self):
        """Test is_known_fp flag when explicitly set."""
        exofop = ExofopDisposition(tic_id="TIC 1", is_known_fp=True)
        assert exofop.is_known_fp is True
        
        exofop = ExofopDisposition(tic_id="TIC 1", is_known_fp=False)
        assert exofop.is_known_fp is False
    
    def test_known_planet_flag_explicit(self):
        """Test is_known_planet flag when explicitly set."""
        exofop = ExofopDisposition(tic_id="TIC 1", is_known_planet=True)
        assert exofop.is_known_planet is True
        
        exofop = ExofopDisposition(tic_id="TIC 1", is_known_planet=False)
        assert exofop.is_known_planet is False


class TestCatalogCache:
    """Tests for CatalogCache."""
    
    def test_cache_write_and_read(self, tmp_path):
        """Test basic cache write and read."""
        cache = CatalogCache(cache_dir=tmp_path, ttl_days=1)
        cache.set("test_source", "test_key", {"value": 42})
        result = cache.get("test_source", "test_key")
        assert result == {"value": 42}
    
    def test_cache_expiry(self, tmp_path):
        """Test that expired cache entries are not returned."""
        cache = CatalogCache(cache_dir=tmp_path, ttl_days=0)  # Immediate expiry
        cache.set("test_source", "test_key", {"value": 42})
        # Manually set old timestamp
        cache_file = tmp_path / "test_source_test_key.pkl"
        import pickle
        with open(cache_file, "wb") as f:
            pickle.dump({"timestamp": time.time() - 86400, "data": {"value": 42}}, f)
        
        result = cache.get("test_source", "test_key")
        assert result is None
    
    def test_cache_missing_key(self, tmp_path):
        """Test that missing keys return None."""
        cache = CatalogCache(cache_dir=tmp_path)
        result = cache.get("nonexistent", "key")
        assert result is None
    
    def test_cache_clear(self, tmp_path):
        """Test clearing all cache entries."""
        cache = CatalogCache(cache_dir=tmp_path)
        cache.set("source1", "key1", {"v": 1})
        cache.set("source2", "key2", {"v": 2})
        # Clear by removing files
        for path in cache.cache_dir.glob("*.pkl"):
            path.unlink()
        assert cache.get("source1", "key1") is None
        assert cache.get("source2", "key2") is None


class TestQueryTIC:
    """Tests for TIC v9 query (mocked)."""
    
    def test_query_tic_v9_success(self):
        """Test successful TIC query - TIC catalog doesn't include GAIA."""
        _clear_global_catalog_cache()
        with patch("anomaly9.catalog.Vizier") as mock_vizier:
            mock_vizier.ROW_LIMIT = 1
            mock_vizier.TIMEOUT = 10
            
            # TIC catalog columns: Teff, logg, rad, mass (no GAIA - that's in query_gaia_dr3)
            mock_table = MagicMock()
            mock_table.__len__ = MagicMock(return_value=1)
            mock_table.colnames = ["Teff", "logg", "rad", "mass"]
            mock_table.__getitem__ = MagicMock(return_value={
                "Teff": 5800,
                "logg": 4.4,
                "rad": 1.2,
                "mass": 1.1,
            })
            mock_vizier.query_constraints.return_value = [mock_table]
            
            result = query_tic_v9("TIC 12345678")
            
            assert result is not None
            assert result.tic_id == "TIC 12345678"
            assert result.radius_solar == 1.2
            assert result.mass_solar == 1.1
            assert result.teff_k == 5800
            assert result.logg == 4.4
            assert result.gaia_dr3_id is None  # TIC query doesn't return GAIA
            assert result.source == "TIC"
    
    def test_query_tic_v9_no_results(self):
        """Test TIC query with no results."""
        _clear_global_catalog_cache()
        with patch("anomaly9.catalog.Vizier") as mock_vizier:
            mock_vizier.query_constraints.return_value = []
            
            result = query_tic_v9("TIC 99999999")
            assert result is None
    
    def test_query_tic_v9_exception_handling(self):
        """Test TIC query handles exceptions gracefully."""
        _clear_global_catalog_cache()
        with patch("anomaly9.catalog.Vizier") as mock_vizier:
            mock_vizier.query_constraints.side_effect = ConnectionError("Network error")
            
            result = query_tic_v9("TIC 12345678")
            # Function catches Exception and returns None
            assert result is None


class TestQueryGaia:
    """Tests for Gaia DR3 cone search (mocked)."""
    
    def test_query_gaia_dr3_by_coords_success(self):
        """Test successful Gaia cone search."""
        _clear_global_catalog_cache()
        with patch("anomaly9.catalog.Gaia") as mock_gaia, \
             patch("anomaly9.catalog.SkyCoord") as mock_skycoord, \
             patch("anomaly9.catalog.u") as mock_u:
            
            mock_gaia.ROW_LIMIT = 1
            mock_gaia.TIMEOUT = 10
            
            mock_job = _make_gaia_result([{
                "source_id": 987654321,
                "ruwe": 1.2,
            }])
            mock_gaia.cone_search_async.return_value = mock_job
            
            result = query_gaia_dr3_by_coords(10.0, 20.0)
            
            assert result is not None
            assert result.gaia_dr3_id == 987654321
            assert result.ruwe == 1.2
            assert result.source == "Gaia"
    
    def test_query_gaia_dr3_by_coords_no_results(self):
        """Test Gaia cone search with no results."""
        _clear_global_catalog_cache()
        with patch("anomaly9.catalog.Gaia") as mock_gaia, \
             patch("anomaly9.catalog.SkyCoord") as mock_skycoord, \
             patch("anomaly9.catalog.u") as mock_u:
            
            mock_gaia.ROW_LIMIT = 1
            mock_gaia.TIMEOUT = 10
            
            mock_job = _make_gaia_result([])
            mock_gaia.cone_search_async.return_value = mock_job
            
            result = query_gaia_dr3_by_coords(10.0, 20.0)
            assert result is None


class TestQueryExoFOP:
    """Tests for ExoFOP-TESS query (mocked)."""
    
    def _clear_global_cache(self):
        """Clear the global cache to avoid test interference."""
        _clear_global_catalog_cache()
    
    @patch("anomaly9.catalog.requests.get")
    def test_query_exofop_success_known_fp(self, mock_get):
        """Test ExoFOP query for known false positive."""
        self._clear_global_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = [{
            "toi": "1234.01",
            "tfopwg_disp": "FP",
            "comments": "Eclipsing binary confirmed"
        }]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        cache = CatalogCache()
        result = query_exofop_tess("TIC 12345678", cache=cache)
        
        assert result is not None
        assert result.tic_id == "TIC 12345678"
        assert result.toi == "1234.01"
        assert result.tfopwg_disp == "FP"
        assert result.is_known_fp is True
        assert result.is_known_planet is False
    
    @patch("anomaly9.catalog.requests.get")
    def test_query_exofop_success_known_planet(self, mock_get):
        """Test ExoFOP query for known planet (CP = both FP and planet in implementation)."""
        self._clear_global_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = [{
            "toi": "5678.01",
            "tfopwg_disp": "CP",
            "comments": "Confirmed planet"
        }]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        cache = CatalogCache()
        result = query_exofop_tess("TIC 56789012", cache=cache)
        
        assert result is not None
        assert result.tfopwg_disp == "CP"
        # CP is treated as both FP and planet in the implementation
        assert result.is_known_planet is True
        assert result.is_known_fp is True  # CP is in FP list too
    
    @patch("anomaly9.catalog.requests.get")
    def test_query_exofop_no_tois(self, mock_get):
        """Test ExoFOP query with no TOI entries."""
        self._clear_global_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        cache = CatalogCache()
        result = query_exofop_tess("TIC 99999999", cache=cache)
        
        # Empty list returns None
        assert result is None
    
    @patch("anomaly9.catalog.requests.get")
    def test_query_exofop_timeout(self, mock_get):
        """Test ExoFOP query timeout."""
        self._clear_global_cache()
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        cache = CatalogCache()
        result = query_exofop_tess("TIC 12345678", cache=cache)
        assert result is None
    
    @patch("anomaly9.catalog.requests.get")
    def test_query_exofop_network_error(self, mock_get):
        """Test ExoFOP query network error."""
        self._clear_global_cache()
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        cache = CatalogCache()
        result = query_exofop_tess("TIC 12345678", cache=cache)
        assert result is None


class TestEnrichWithCatalogs:
    """Tests for enrich_with_catalogs integration."""
    
    def test_enrich_with_catalogs_disabled(self):
        """Test enrichment with all catalogs disabled."""
        lc_data = LightCurveData(
            target="TIC 12345678",
            mission="TESS",
            time=np.array([1.0, 2.0, 3.0]),
            raw_flux=np.array([1.0, 1.0, 1.0]),
            flux=np.array([1.0, 1.0, 1.0]),
        )
        
        enrich_with_catalogs(lc_data, enable_gaia=False, enable_tic=False, enable_exofop=False)
        
        assert lc_data.metadata.get("stellar_metadata") is None
        assert lc_data.metadata.get("exofop_disposition") is None
    
    @patch("anomaly9.catalog.query_tic_v9")
    @patch("anomaly9.catalog.query_exofop_tess")
    def test_enrich_with_catalogs_tic_only(self, mock_exofop, mock_tic):
        """Test enrichment with TIC only."""
        _clear_global_catalog_cache()
        mock_tic.return_value = StellarMetadata(
            tic_id="TIC 12345678",
            radius_solar=1.0,
            mass_solar=1.0,
        )
        mock_exofop.return_value = None
        
        lc_data = LightCurveData(
            target="TIC 12345678",
            mission="TESS",
            time=np.array([1.0, 2.0, 3.0]),
            raw_flux=np.array([1.0, 1.0, 1.0]),
            flux=np.array([1.0, 1.0, 1.0]),
        )
        
        enrich_with_catalogs(lc_data, enable_gaia=False, enable_tic=True, enable_exofop=False)
        
        assert lc_data.metadata.get("stellar_metadata") is not None
        assert lc_data.metadata["stellar_metadata"].radius_solar == 1.0
        assert lc_data.metadata.get("exofop_disposition") is None
    
    @patch("anomaly9.catalog.query_tic_v9")
    @patch("anomaly9.catalog.query_exofop_tess")
    def test_enrich_with_catalogs_exofop_fp(self, mock_exofop, mock_tic):
        """Test enrichment with ExoFOP returning known FP."""
        _clear_global_catalog_cache()
        mock_tic.return_value = None
        mock_exofop.return_value = ExofopDisposition(
            tic_id="TIC 12345678",
            tfopwg_disp="FP",
            is_known_fp=True,
        )
        
        lc_data = LightCurveData(
            target="TIC 12345678",
            mission="TESS",
            time=np.array([1.0, 2.0, 3.0]),
            raw_flux=np.array([1.0, 1.0, 1.0]),
            flux=np.array([1.0, 1.0, 1.0]),
        )
        
        enrich_with_catalogs(lc_data, enable_gaia=False, enable_tic=False, enable_exofop=True)
        
        assert lc_data.metadata.get("stellar_metadata") is None
        assert lc_data.metadata.get("exofop_disposition") is not None
        assert lc_data.metadata["exofop_disposition"].is_known_fp is True


class TestHelperFunctions:
    """Tests for catalog helper functions."""
    
    def test_get_stellar_density(self):
        """Test get_stellar_density helper."""
        lc_data = LightCurveData(
            target="TIC 1",
            mission="TESS",
            time=np.array([1.0, 2.0, 3.0]),
            raw_flux=np.array([1.0, 1.0, 1.0]),
            flux=np.array([1.0, 1.0, 1.0]),
        )
        lc_data.metadata["stellar_metadata"] = StellarMetadata(
            tic_id="TIC 1", mass_solar=1.0, radius_solar=1.0
        )
        
        density = get_stellar_density(lc_data)
        assert density is not None
        assert abs(density - 1.408) < 0.01
    
    def test_get_stellar_density_none(self):
        """Test get_stellar_density when no metadata."""
        lc_data = LightCurveData(
            target="TIC 1",
            mission="TESS",
            time=np.array([1.0, 2.0, 3.0]),
            raw_flux=np.array([1.0, 1.0, 1.0]),
            flux=np.array([1.0, 1.0, 1.0]),
        )
        
        density = get_stellar_density(lc_data)
        assert density is None
    
    def test_get_stellar_radius(self):
        """Test get_stellar_radius helper."""
        lc_data = LightCurveData(
            target="TIC 1",
            mission="TESS",
            time=np.array([1.0, 2.0, 3.0]),
            raw_flux=np.array([1.0, 1.0, 1.0]),
            flux=np.array([1.0, 1.0, 1.0]),
        )
        lc_data.metadata["stellar_metadata"] = StellarMetadata(
            tic_id="TIC 1", radius_solar=1.5
        )
        
        radius = get_stellar_radius(lc_data)
        assert radius == 1.5
    
    def test_is_known_fp(self):
        """Test is_known_fp helper."""
        lc_data = LightCurveData(
            target="TIC 1",
            mission="TESS",
            time=np.array([1.0, 2.0, 3.0]),
            raw_flux=np.array([1.0, 1.0, 1.0]),
            flux=np.array([1.0, 1.0, 1.0]),
        )
        lc_data.metadata["exofop_disposition"] = ExofopDisposition(
            tic_id="TIC 1", is_known_fp=True
        )
        
        assert is_known_fp(lc_data) is True
        
        lc_data.metadata["exofop_disposition"] = ExofopDisposition(
            tic_id="TIC 1", is_known_fp=False
        )
        assert is_known_fp(lc_data) is False
        
        lc_data.metadata["exofop_disposition"] = None
        assert is_known_fp(lc_data) is False
    
    def test_is_known_planet(self):
        """Test is_known_planet helper."""
        lc_data = LightCurveData(
            target="TIC 1",
            mission="TESS",
            time=np.array([1.0, 2.0, 3.0]),
            raw_flux=np.array([1.0, 1.0, 1.0]),
            flux=np.array([1.0, 1.0, 1.0]),
        )
        lc_data.metadata["exofop_disposition"] = ExofopDisposition(
            tic_id="TIC 1", is_known_planet=True
        )
        
        assert is_known_planet(lc_data) is True
        
        lc_data.metadata["exofop_disposition"] = ExofopDisposition(
            tic_id="TIC 1", is_known_planet=False
        )
        assert is_known_planet(lc_data) is False
    
    def test_get_exofop_disposition(self):
        """Test get_exofop_disposition helper."""
        lc_data = LightCurveData(
            target="TIC 1",
            mission="TESS",
            time=np.array([1.0, 2.0, 3.0]),
            raw_flux=np.array([1.0, 1.0, 1.0]),
            flux=np.array([1.0, 1.0, 1.0]),
        )
        lc_data.metadata["exofop_disposition"] = ExofopDisposition(
            tic_id="TIC 1", tfopwg_disp="FP"
        )
        
        assert get_exofop_disposition(lc_data) == "FP"
        
        lc_data.metadata["exofop_disposition"] = None
        assert get_exofop_disposition(lc_data) is None


class TestDurationPeriodConsistency:
    """Tests for duration_period_consistency with physical stellar density."""
    
    def test_physical_density_consistent(self):
        """Test that physical density gives expected duration range."""
        # Sun-like star: rho = 1.408 g/cm³
        # Period = 10 days
        # Expected a/R* ≈ 17.5, duration/period ≈ 0.018
        result = duration_period_consistency(
            period=10.0,
            duration=0.18,  # 0.018 * 10
            depth_ppm=1000,
            stellar_density_gcc=1.408,
        )
        assert result.passed is True
    
    def test_physical_density_too_short(self):
        """Test that too-short duration fails with physical density."""
        # Very short duration for given density
        result = duration_period_consistency(
            period=10.0,
            duration=0.01,  # Too short
            depth_ppm=1000,
            stellar_density_gcc=1.408,
        )
        assert result.passed is False
        assert "0.5" in result.description
    
    def test_physical_density_too_long(self):
        """Test that too-long duration fails with physical density."""
        result = duration_period_consistency(
            period=10.0,
            duration=5.0,  # Too long (50% of period!)
            depth_ppm=1000,
            stellar_density_gcc=1.408,
        )
        assert result.passed is False
        assert "2" in result.description
    
    def test_fallback_heuristic_when_no_density(self):
        """Test fallback to heuristic when no density provided."""
        # Without density, uses heuristic limits 0.1%-10%
        result = duration_period_consistency(
            period=10.0,
            duration=0.05,  # 0.5% - within heuristic range
            depth_ppm=1000,
            stellar_density_gcc=None,
        )
        assert result.passed is True
    
    def test_fallback_heuristic_flags_extreme(self):
        """Test fallback heuristic flags extreme ratios."""
        result = duration_period_consistency(
            period=10.0,
            duration=0.005,  # 0.05% - below 0.1% heuristic minimum
            depth_ppm=1000,
            stellar_density_gcc=None,
        )
        assert result.passed is False
        assert "0.001" in result.description
    
    def test_deep_short_transit_flagged(self):
        """Test that deep + short transits are flagged as grazing EB."""
        # No density, but deep and short
        result = duration_period_consistency(
            period=10.0,
            duration=0.02,  # 0.2% of period
            depth_ppm=20000,  # 2% depth - very deep
            stellar_density_gcc=None,
        )
        assert result.passed is False
        assert "grazing EB" in result.description


class TestIntegration:
    """Integration tests for catalog + vetting pipeline."""
    
    @patch("anomaly9.catalog.query_tic_v9")
    @patch("anomaly9.catalog.query_exofop_tess")
    def test_full_enrichment_and_vetting(self, mock_exofop, mock_tic):
        """Test full pipeline: catalog enrichment -> vetting with physical density."""
        _clear_global_catalog_cache()
        # Mock TIC returning solar-like star
        mock_tic.return_value = StellarMetadata(
            tic_id="TIC 12345678",
            radius_solar=1.0,
            mass_solar=1.0,
            teff_k=5778,
            logg=4.44,
        )
        # Mock ExoFOP returning no known FP
        mock_exofop.return_value = ExofopDisposition(
            tic_id="TIC 12345678",
            is_known_fp=False,
        )
        
        lc_data = LightCurveData(
            target="TIC 12345678",
            mission="TESS",
            time=np.arange(1000, dtype=float),
            raw_flux=np.ones(1000),
            flux=np.ones(1000),
        )
        
        enrich_with_catalogs(lc_data, enable_gaia=False, enable_tic=True, enable_exofop=True)
        
        # Check metadata was enriched
        assert lc_data.metadata.get("stellar_metadata") is not None
        assert lc_data.metadata.get("exofop_disposition") is not None
        
        # Get stellar density
        density = get_stellar_density(lc_data)
        assert density is not None
        assert abs(density - 1.408) < 0.01
        
        # Run vetting check with physical density
        result = duration_period_consistency(
            period=10.0,
            duration=0.18,
            depth_ppm=1000,
            stellar_density_gcc=density,
        )
        
        # Should pass with physical density
        assert result.passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])