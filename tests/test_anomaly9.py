"""
Unit tests for Anomaly9 Phase 1 engine components.
"""

import numpy as np
import pytest

from anomaly9.pipeline import clean_lightcurve, detrend_lightcurve, LightCurveData
from anomaly9.detector import sliding_zscore, run_isolation_forest, cluster_anomalies, run_dual_detector


def test_clean_lightcurve():
    time = np.linspace(0, 10, 100)
    flux = np.ones(100)
    # inject NaN and negative value
    flux[10] = np.nan
    flux[20] = -1.0
    # inject outlier
    flux[50] = 50.0

    t_cl, f_cl, fe_cl, n_rem = clean_lightcurve(time, flux, sigma_clip=5.0)
    assert n_rem >= 3
    assert not np.isnan(f_cl).any()
    assert (f_cl > 0).all()


def test_detrend_lightcurve():
    time = np.linspace(0, 5, 200)
    # slow stellar variability trend + synthetic transit dip
    trend = 1.0 + 0.05 * np.sin(2 * np.pi * time / 2.0)
    dip = np.zeros_like(time)
    dip[(time >= 2.2) & (time <= 2.4)] = -0.1
    flux = trend + dip

    detrended = detrend_lightcurve(time, flux, window_savgol=51, poly_order=3)
    assert detrended.size == time.size
    # Dip should be preserved as a negative anomaly in detrended flux
    assert np.min(detrended) < 0.95


def test_sliding_zscore():
    flux = np.ones(100)
    flux[50] = 0.5  # dip
    z = sliding_zscore(flux, window=11)
    assert z.size == 100
    assert z[50] < -2.0


def test_detector_pipeline():
    # Synthetic light curve with a periodic transit signal
    n = 300
    time = np.linspace(0, 6, n)
    period = 1.5
    flux = np.ones(n)
    # inject transits at t = 1.5, 3.0, 4.5
    for epoch in [1.5, 3.0, 4.5]:
        in_t = (time >= epoch - 0.05) & (time <= epoch + 0.05)
        flux[in_t] = 0.98

    lc_data = LightCurveData(
        target="SyntheticTest",
        mission="TESS",
        time=time,
        raw_flux=flux,
        flux=flux / np.median(flux),
        cadence_s=120.0,
    )

    report = run_dual_detector(lc_data, period_min_d=1.0, period_max_d=2.5, period_step_d=0.01)
    assert report.bls_best is not None
    # Best period should be close to 1.5 days
    assert abs(report.bls_best.period_d - 1.5) < 0.1
