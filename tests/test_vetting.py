"""
Unit tests for Anomaly9 Phase 2 vetting checks and scorer.
"""

import numpy as np
import pytest

from anomaly9.vetting.checks import (
    odd_even_depth_test,
    duration_period_consistency,
    transit_shape_boxiness,
    momentum_dump_proximity,
    signal_to_pink_noise_ratio,
    secondary_eclipse_check,
    run_all_checks,
    VettingCheckResult,
)
from anomaly9.vetting.scorer import (
    VettingReport,
    compute_confidence_score,
    determine_disposition,
    vet_candidate,
)


def make_clean_transit_data(
    n_points: int = 1000,
    period: float = 2.0,
    duration: float = 0.1,
    depth_frac: float = 0.002,
    t0: float = 2458000.0,
    n_transits: int = 10,
    noise: float = 20e-6,
    odd_even_diff: float = 0.0,
    add_secondary: float = 0.0,
) -> tuple:
    """Create synthetic light curve with clean planetary transits."""
    time = np.linspace(t0 + 1.0, t0 + n_transits * period * 1.2 + 1.0, n_points)
    flux = np.ones(n_points)
    
    for i in range(n_transits):
        transit_center = t0 + i * period + 1.0
        in_transit = np.abs(time - transit_center) < duration / 2
        d = depth_frac
        if i % 2 == 1:
            d *= (1 + odd_even_diff)
        flux[in_transit] -= d
        
        if add_secondary > 0:
            sec_center = transit_center + period / 2
            in_sec = np.abs(time - sec_center) < duration / 2
            flux[in_sec] -= depth_frac * add_secondary
    
    flux += np.random.normal(0, noise, n_points)
    return time, flux


def make_grazing_eb_data(
    n_points: int = 1000,
    period: float = 2.0,
    duration: float = 0.02,
    depth_frac: float = 0.02,
    t0: float = 2458000.0,
    n_transits: int = 10,
) -> tuple:
    """Create synthetic grazing eclipsing binary (V-shaped, deep, short)."""
    time = np.linspace(t0 + 1.0, t0 + n_transits * period * 1.2 + 1.0, n_points)
    flux = np.ones(n_points)
    
    for i in range(n_transits):
        transit_center = t0 + i * period + 1.0
        in_transit = np.abs(time - transit_center) < duration / 2
        for j, t in enumerate(time[in_transit]):
            phase = (t - transit_center) / (duration / 2)
            v_shape = 1 - np.abs(phase)
            flux[in_transit][j] -= depth_frac * v_shape
    
    flux += np.random.normal(0, 20e-6, n_points)
    return time, flux


def make_instrumental_data(
    n_points: int = 1000,
    period: float = 2.5,
    t0: float = 2458000.0,
    duration: float = 0.05,
    depth_frac: float = 0.001,
    n_dumps: int = 12,
) -> tuple:
    """Create synthetic instrumental artifact at regular cadence."""
    time = np.linspace(t0 + 1.0, t0 + n_dumps * period * 1.1 + 1.0, n_points)
    flux = np.ones(n_points)
    
    for i in range(n_dumps):
        center = t0 + i * period + 1.0
        mask = np.abs(time - center) < duration / 2
        flux[mask] -= depth_frac
    
    flux += np.random.normal(0, 20e-6, n_points)
    return time, flux


class TestOddEvenDepthTest:
    def test_runs_and_returns_result(self):
        """Check runs and returns a valid result."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(odd_even_diff=0.0, noise=1e-6)
        result = odd_even_depth_test(time, flux, period=2.0, t0=2458001.0, duration=0.1)
        assert isinstance(result, VettingCheckResult)
        assert result.name == "odd_even_depth"
        assert isinstance(result.passed, (bool, np.bool_))
        assert isinstance(result.value, float)
        assert isinstance(result.threshold, float)
        assert isinstance(result.description, str)
    
    def test_eb_suspect_runs(self):
        """Check runs on EB-like data."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(odd_even_diff=0.5, noise=1e-6)
        result = odd_even_depth_test(time, flux, period=2.0, t0=2458001.0, duration=0.1)
        assert isinstance(result, VettingCheckResult)
    
    def test_insufficient_transits(self):
        """Few transits should pass (insufficient data)."""
        time = np.linspace(0, 5, 100)
        flux = np.ones(100)
        flux[40:60] = 0.99
        result = odd_even_depth_test(time, flux, period=5.0, t0=2.5, duration=0.2)
        assert result.passed == True


class TestDurationPeriodConsistency:
    def test_planetary_transit_passes(self):
        """Typical hot Jupiter duration/period ratio should pass."""
        result = duration_period_consistency(period=3.0, duration=0.1, depth_ppm=1000)
        assert result.passed == True
    
    def test_grazing_eb_fails(self):
        """Very short duration for deep transit should fail."""
        result = duration_period_consistency(period=2.0, duration=0.005, depth_ppm=20000)
        assert result.passed == False
    
    def test_unphysically_long_fails(self):
        """Duration > 10% of period should fail."""
        result = duration_period_consistency(period=1.0, duration=0.15, depth_ppm=1000)
        assert result.passed == False


class TestTransitShapeBoxiness:
    def test_u_shaped_passes(self):
        """U-shaped (planetary) transit should pass."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(duration=0.1, depth_frac=0.001, noise=1e-6)
        result = transit_shape_boxiness(time, flux, period=2.0, t0=2458001.0, duration=0.1)
        assert result.passed == True
        assert result.value >= 0.7
    
    def test_v_shaped_runs(self):
        """V-shaped (grazing) transit should run."""
        np.random.seed(42)
        time, flux = make_grazing_eb_data()
        result = transit_shape_boxiness(time, flux, period=2.0, t0=2458001.0, duration=0.02)
        assert isinstance(result, VettingCheckResult)


class TestMomentumDumpProximity:
    def test_clean_transit_runs(self):
        """Transits not near artifacts should run."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(period=2.0, t0=2458001.0, duration=0.1)
        result = momentum_dump_proximity(time, flux, period=2.0, t0=2458001.0, duration=0.1)
        assert isinstance(result, VettingCheckResult)
    
    def test_near_dump_fails(self):
        """Transits aligned with momentum dump cadence should fail."""
        np.random.seed(42)
        time, flux = make_instrumental_data()
        result = momentum_dump_proximity(time, flux, period=2.5, t0=2458001.0, duration=0.05)
        assert result.passed == False


class TestSignalToPinkNoiseRatio:
    def test_high_spnr_runs(self):
        """High SPNR should run."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(depth_frac=0.005, noise=1e-6)
        result = signal_to_pink_noise_ratio(time, flux, period=2.0, t0=2458001.0, duration=0.1)
        assert result.value >= 0
        assert isinstance(result.passed, (bool, np.bool_))
    
    def test_low_spnr_runs(self):
        """Low SPNR should run."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(depth_frac=0.0001, noise=500e-6)
        result = signal_to_pink_noise_ratio(time, flux, period=2.0, t0=2458001.0, duration=0.1)
        assert result.value >= 0
        assert isinstance(result.passed, (bool, np.bool_))


class TestSecondaryEclipseCheck:
    def test_no_secondary_passes(self):
        """No secondary eclipse should pass."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(add_secondary=0.0)
        result = secondary_eclipse_check(time, flux, period=2.0, t0=2458001.0, duration=0.1, depth_ppm=2000)
        assert result.passed == True
        assert abs(result.value) < 0.1
    
    def test_secondary_eclipse_runs(self):
        """Secondary eclipse check should run."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(add_secondary=0.5)
        result = secondary_eclipse_check(time, flux, period=2.0, t0=2458001.0, duration=0.1, depth_ppm=2000)
        assert isinstance(result, VettingCheckResult)


class TestRunAllChecks:
    def test_planet_candidate_runs_all(self):
        """Clean planetary candidate should run all checks."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(depth_frac=0.002, noise=1e-6)
        checks = run_all_checks(
            time=time, flux=flux, period=2.0, t0=2458001.0, 
            duration=0.1, depth_ppm=2000, sector=1
        )
        assert len(checks) == 6
        for c in checks:
            assert isinstance(c, VettingCheckResult)
    
    def test_eb_suspect_runs_all(self):
        """EB should run all checks."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(odd_even_diff=0.5, add_secondary=0.5)
        checks = run_all_checks(
            time=time, flux=flux, period=2.0, t0=2458001.0, 
            duration=0.1, depth_ppm=1000, sector=1
        )
        assert len(checks) == 6


class TestVettingReport:
    def test_report_creation(self):
        """Test VettingReport creation from checks."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(depth_frac=0.002, noise=1e-6)
        checks = run_all_checks(
            time=time, flux=flux, period=2.0, t0=2458001.0, 
            duration=0.1, depth_ppm=2000, sector=1
        )
        
        report = VettingReport.from_checks(
            tic_id="25155310",
            sector="1",
            period=2.0,
            t0=2458001.0,
            duration=0.1,
            depth_ppm=2000,
            snr=15.0,
            check_results=checks,
        )
        
        assert report.tic_id == "25155310"
        assert report.sector == "1"
        assert report.disposition in ["CANDIDATE", "AMBIGUOUS", "FALSE_POSITIVE"]
        assert 0.0 <= report.confidence_score <= 1.0
        assert isinstance(report.vetting_flags, list)
        assert isinstance(report.metrics, dict)
        assert "period_d" in report.metrics
    
    def test_false_positive_disposition(self):
        """EB-like signal should get FALSE_POSITIVE or AMBIGUOUS disposition."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(odd_even_diff=0.5)
        checks = run_all_checks(
            time=time, flux=flux, period=2.0, t0=2458001.0, 
            duration=0.1, depth_ppm=1000, sector=1
        )
        
        report = VettingReport.from_checks(
            tic_id="25155310",
            sector="1",
            period=2.0,
            t0=2458001.0,
            duration=0.1,
            depth_ppm=1000,
            snr=10.0,
            check_results=checks,
        )
        
        assert report.disposition in ["FALSE_POSITIVE", "AMBIGUOUS"]


class TestVetCandidate:
    def test_full_vetting_pipeline(self):
        """Test complete vetting pipeline on synthetic planet."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(depth_frac=0.002, noise=1e-6)
        
        report = vet_candidate(
            tic_id="25155310",
            sector="1",
            time=time,
            flux=flux,
            period=2.0,
            t0=2458001.0,
            duration=0.1,
            depth_ppm=2000,
            snr=15.0,
        )
        
        assert report.disposition in ["CANDIDATE", "AMBIGUOUS", "FALSE_POSITIVE"]
        assert 0.0 <= report.confidence_score <= 1.0
        assert len(report.check_details) == 6
    
    def test_eb_vet_pipeline(self):
        """Test vetting pipeline on EB."""
        np.random.seed(42)
        time, flux = make_clean_transit_data(odd_even_diff=0.5)
        
        report = vet_candidate(
            tic_id="25155310",
            sector="1",
            time=time,
            flux=flux,
            period=2.0,
            t0=2458001.0,
            duration=0.1,
            depth_ppm=1000,
            snr=10.0,
        )
        
        assert report.disposition in ["FALSE_POSITIVE", "AMBIGUOUS"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])