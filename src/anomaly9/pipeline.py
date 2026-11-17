"""
Anomaly9 Pipeline
=================
Core NASA MAST ingestion: MAST search, FITS download, quality-flag / NaN
cleaning, outlier clipping, and Savitzky-Golay detrending that preserves
transit-shaped features while removing stellar & instrumental noise.
"""

from __future__ import annotations

import logging
import time as time_mod
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
import astroquery.mast

# Configure generous timeout for NASA MAST server queries and downloads
astroquery.mast.Conf.timeout = 300

# Enable AWS S3 cloud dataset streaming so FITS files stream reliably
# from NASA's MAST AWS backend instead of dropping mid-download over HTTP.
from astroquery.mast import Observations
try:
    Observations.enable_cloud_dataset()
except Exception as _cloud_exc:  # pragma: no cover - boto3/credentials may be absent
    logging.getLogger("anomaly9.pipeline").warning(
        "MAST AWS S3 cloud dataset unavailable (%s); falling back to standard MAST HTTP downloads.",
        _cloud_exc,
    )

logger = logging.getLogger("anomaly9.pipeline")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from lightkurve import search_lightcurve, LightCurveCollection, read as lk_read
    try:
        from lightkurve.exceptions import LightkurveError
    except ImportError:
        LightkurveError = Exception

__all__ = [
    "LightCurveData",
    "search_mast",
    "ingest_mast_lightcurve",
    "clean_lightcurve",
    "detrend_lightcurve",
    "normalize_target_id",
    "sanitize_filename",
]


def _download_fits_chunked(url: str, cache_dir: Path, max_attempts: int = 3) -> Path:
    """Download a FITS file from a MAST URL using chunked, resumable HTTP requests.

    Uses ``requests`` with ``Range`` header to support resume of interrupted downloads.
    Streams to ``cache_dir/<filename>.fits`` with ``stream=True``.
    """
    filename = url.split("/")[-1]
    if not filename.endswith(".fits"):
        filename = filename + ".fits"
    local_path = cache_dir / filename
    headers = {}
    if local_path.exists():
        existing_size = local_path.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"
    else:
        existing_size = 0

    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=astroquery.mast.Conf.timeout) as r:
                r.raise_for_status()
                mode = "ab" if existing_size > 0 else "wb"
                with open(local_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            logger.info("Downloaded %s (%d bytes)", local_path, local_path.stat().st_size)
            return local_path
        except (requests.exceptions.RequestException, OSError) as e:
            logger.warning("Chunked download attempt %d/%d failed: %s", attempt, max_attempts, e)
            if local_path.exists():
                try:
                    local_path.unlink()
                except Exception:
                    pass
            if attempt == max_attempts:
                raise
            time_mod.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url} after {max_attempts} attempts")


@dataclass
class LightCurveData:
    """Container for a cleaned & detrended MAST light curve."""

    target: str
    mission: str
    time: np.ndarray          # BJD (TDB) or Kepler mission time
    raw_flux: np.ndarray
    flux: np.ndarray          # detrended (normalized) flux
    flux_err: Optional[np.ndarray] = None
    cadence_s: float = 60.0
    duration_days: float = 0.0
    n_points: int = 0
    n_removed: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_points = int(self.time.size)
        self.duration_days = float(self.time[-1] - self.time[0]) if self.time.size else 0.0
        if self.cadence_s <= 0 and self.time.size > 1:
            self.cadence_s = float(np.median(np.diff(self.time))) * 86400.0


# --------------------------------------------------------------------------
# MAST search & download
# --------------------------------------------------------------------------

def search_mast(target: str, mission: str = "TESS") -> List[dict]:
    """Query NASA MAST (via lightkurve) for available light curves.

    Parameters
    ----------
    target : str
        TIC, KIC, RA/Dec, or target identifier.
    mission : str
        'TESS' or 'Kepler'.

    Returns
    -------
    list of dicts describing candidate light curves (download_url, ...).
    """
    mission = mission.lower()
    author = "SPOC" if mission == "tess" else "Kepler"
    logger.info("Querying NASA MAST for %r (mission=%s, author=%s)", target, mission, author)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lc_search = search_lightcurve(target, mission=mission, author=author, limit=50, cloud_only=True)
        except TypeError:
            lc_search = search_lightcurve(target, mission=mission, author=author, limit=50)

    # Fallback to unfiltered author search if official author yields nothing
    if lc_search is None or len(lc_search) == 0:
        logger.info("Author %r returned 0 results; retrying without author filter.", author)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lc_search = search_lightcurve(target, mission=mission, limit=50, cloud_only=True)
            except TypeError:
                lc_search = search_lightcurve(target, mission=mission, limit=50)

    if lc_search is None or len(lc_search) == 0:
        raise LookupError(
            f"No light curves found in MAST for target {target!r} (mission={mission}). "
            "Check the TIC/KIC identifier and try the other mission."
        )

    candidates = []
    for i in range(len(lc_search)):
        lc = lc_search[i]
        try:
            candidates.append(
                {
                    "index": i,
                    "target_name": str(getattr(lc, "target_name", target)),
                    "mission": str(getattr(lc, "mission", mission.upper())),
                    "sector": str(getattr(lc, "sector", "") or ""),
                    "quarter": str(getattr(lc, "quarter", "") or ""),
                    "cadence_min": getattr(lc, "cadence_min", None),
                    "download_url": getattr(lc, "download_url", None),
                }
            )
        except Exception:  # pragma: no cover - defensive
            continue

    logger.info("Found %d candidate light curve(s) in MAST.", len(candidates))
    return candidates


def ingest_mast_lightcurve(
    target: str,
    mission: str = "TESS",
    quality: str = "cleaned",
    sector: Optional[int] = None,
    quarter: Optional[int] = None,
    cadence_min: int = 180,
    window_savgol: int = 120,
    poly_order: int = 3,
    sigma_clip: float = 6.0,
    fits_file: Optional[str] = None,
) -> LightCurveData:
    """Full Phase-1 ingestion pipeline.

    Steps:
      1. If ``fits_file`` is provided, load it directly via ``lk.read`` without
         hitting the network; otherwise search MAST for the target.
      2. Download a single observation FITS light curve into data/cache with a
         robust retry loop (up to 3 attempts, clearing partial files).
      3. Clean: drop NaN/inf, negative flux, saturated & cosmic-ray flagged
         points; sigma-clip gross outliers.
      4. Normalize and detrend with a Savitzky-Golay filter.
    """
    if fits_file is not None:
        lc = lk_read(fits_file)
    else:
        candidates = search_mast(target, mission)

        # Pick the shortest sector (small FITS download, fast iteration).
        if sector is not None:
            candidates = [c for c in candidates if c.get("sector") == sector or c.get("quarter") == sector]
            if not candidates:
                raise LookupError(f"Requested sector/quarter {sector} not available for {target!r}.")
        candidates.sort(key=lambda c: (c.get("cadence_min") or 1e9))
        chosen = candidates[0]
        logger.info(
            "Selected: sector=%s quarter=%s (~%s min) url=%s",
            chosen.get("sector"), chosen.get("quarter"), chosen.get("cadence_min"),
            (chosen.get("download_url") or "")[:80],
        )

        mission_lower = mission.lower()
        author = "SPOC" if mission_lower == "tess" else "Kepler"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lc_search = search_lightcurve(target, mission=mission_lower, author=author, limit=50, cloud_only=True)
            except TypeError:
                lc_search = search_lightcurve(target, mission=mission_lower, author=author, limit=50)
            if lc_search is None or len(lc_search) == 0:
                try:
                    lc_search = search_lightcurve(target, mission=mission_lower, limit=50, cloud_only=True)
                except TypeError:
                    lc_search = search_lightcurve(target, mission=mission_lower, limit=50)

        # Re-locate the exact search row we chose (stable by URL).
        lc_obj = None
        for i in range(len(lc_search)):
            lc = lc_search[i]
            if getattr(lc, "download_url", None) == chosen["download_url"]:
                lc_obj = lc
                break
        if lc_obj is None:
            lc_obj = lc_search[0]

        flux_column = "SAP_FLUX" if "TESS" in str(mission).upper() else "PDCSAP_FLUX"
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the FITS data URI: check table columns in exact order (dataURL, then dataURI),
        # with fallback to Observations.get_product_list if neither is present.
        data_uri = None
        row_idx = i if 'i' in locals() and lc_search.table is not None and i < len(lc_search.table) else 0
        try:
            if lc_search.table is not None and len(lc_search.table) > row_idx:
                row = lc_search.table[row_idx]
                for col in ["dataURL", "dataURI"]:
                    if col in row.colnames:
                        val = row[col]
                        if val:
                            data_uri = str(val)
                            break
        except Exception:
            pass

        if not data_uri:
            data_uri = chosen.get("download_url") or getattr(lc_obj, "download_url", None)

        if not data_uri:
            try:
                sub_table = lc_search.table[row_idx:row_idx+1] if lc_search.table is not None else None
                if sub_table is not None and len(sub_table) > 0:
                    products = Observations.get_product_list(sub_table)
                    if products is not None and len(products) > 0 and "dataURI" in products.colnames:
                        p_uri = products[0]["dataURI"]
                        if p_uri:
                            data_uri = str(p_uri)
                            url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={data_uri}"
            except Exception:
                pass

        if not data_uri:
            raise RuntimeError(f"Cannot determine FITS download URI for target {target!r}.")

        if data_uri.startswith("http"):
            url = data_uri
        else:
            url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={data_uri}"

        lc = None
        for attempt in range(1, 4):
            try:
                logger.info("Chunked download of FITS light curve (attempt %d/3) to %s ...", attempt, cache_dir)
                local_path = _download_fits_chunked(url, cache_dir, max_attempts=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lc = lk_read(str(local_path))
                if lc is not None:
                    break
            except Exception as e:
                logger.warning("Download attempt %d/3 failed: %s", attempt, e)
                # Remove any partial or corrupt .fits files from cache
                for f_path in cache_dir.rglob("*.fits*"):
                    try:
                        f_path.unlink()
                        logger.debug("Cleaned partial file: %s", f_path)
                    except Exception:
                        pass
                if attempt == 3:
                    raise RuntimeError(
                        f"Failed to download FITS data for {target!r} after 3 attempts due to network errors."
                    ) from e
                time_mod.sleep(2 * attempt)

    if isinstance(lc, LightCurveCollection):
        lc = lc[0]

    time = np.asarray(lc.time.value, dtype=np.float64)
    flux = np.asarray(lc.flux.value, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        flux_err = np.asarray(lc.flux_err.value, dtype=np.float64) if lc.flux_err is not None else None

    metadata = {
        "target_name": str(getattr(lc, "target_name", target)),
        "mission": str(getattr(lc, "mission", mission)),
        "sector": str(getattr(lc, "sector", "")),
        "quarter": str(getattr(lc, "quarter", "")),
        "quality": str(getattr(lc, "quality", "")),
        "flux_column": "SAP_FLUX" if "TESS" in str(mission).upper() else "PDCSAP_FLUX",
        "teff": str(getattr(lc, "teff", "")),
        "t0_bjd": str(getattr(lc, "t0_bjd", "")),
        "period": str(getattr(lc, "period", "")),
    }
    logger.info(
        "Downloaded %d points; duration %.2f d; metadata: sector=%s",
        time.size, (time[-1] - time[0]) if time.size else 0.0, metadata["sector"],
    )

    cleaned_time, cleaned_flux, cleaned_err, n_removed = clean_lightcurve(
        time, flux, flux_err, lc=lc, sigma_clip=sigma_clip
    )
    logger.info("Cleaning removed %d / %d points.", n_removed, time.size)

    flux_norm = detrend_lightcurve(cleaned_time, cleaned_flux, window_savgol=window_savgol, poly_order=poly_order)

    cadence_s = float(np.median(np.diff(cleaned_time))) * 86400.0 if cleaned_time.size > 1 else 60.0
    return LightCurveData(
        target=target,
        mission=mission.upper(),
        time=cleaned_time,
        raw_flux=cleaned_flux,
        flux=flux_norm,
        flux_err=cleaned_err,
        cadence_s=cadence_s,
        n_removed=int(n_removed),
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    lc=None,
    sigma_clip: float = 6.0,
) -> tuple:
    """Remove NaN/inf, non-finite errors, saturated/cr-flagged & gross outliers.

    Returns (time, flux, flux_err, n_removed).
    """
    n0 = time.size
    mask = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    if flux_err is not None:
        mask &= np.isfinite(flux_err)

    # Mask points flagged by the archive quality bitmask (saturated / CR / occulting wheel, etc.)
    if lc is not None and hasattr(lc, "quality_bitmask"):
        qm = np.asarray(lc.quality_bitmask)
        if qm.size == n0:
            mask &= qm == 0

    t, f, fe = time[mask], flux[mask], (flux_err[mask] if flux_err is not None else None)

    # Sigma-clipping gross outliers (instrumental spikes), not transits:
    # transits are narrow & deep; a 6-sigma clip on the median preserves them.
    if f.size > 20 and sigma_clip > 0:
        med = np.median(f)
        mad = np.median(np.abs(f - med)) * 1.4826
        mad = max(mad, 1e-3 * abs(med), 1e-4)  # robust MAD floor against zero-MAD constant arrays
        sigma_mask = np.abs(f - med) / mad <= sigma_clip
        n_clipped = int((~sigma_mask).sum())
        t, f, fe = t[sigma_mask], f[sigma_mask], (fe[sigma_mask] if fe is not None else None)
        logger.debug("Sigma-clipped %d gross outliers.", n_clipped)

    n_removed = int(n0 - t.size)
    return t, f, (fe if fe is not None else np.ones_like(f)), n_removed


# --------------------------------------------------------------------------
# Detrending
# --------------------------------------------------------------------------

def detrend_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    window_savgol: int = 120,
    poly_order: int = 3,
    smooth_window_days: float = 2.0,
) -> np.ndarray:
    """Savitzky-Golay detrend.

    A window much longer than any expected transit duration is applied so the
    filter tracks only the slowly varying stellar/systematics trend; transits
    survive in the residual. The window is scaled to the sampling cadence to
    cover ~smooth_window_days by default, with a hard minimum.
    """
    n = flux.size
    if n < 10:
        return flux / np.median(flux)

    cadence_d = (time[-1] - time[0]) / max(n - 1, 1)
    span = max(int(round(smooth_window_days / cadence_d)), 15)
    win = max(span, int(window_savgol))
    # Savitzky-Golay requires odd window, poly_order < window, window <= n.
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 5:
        return flux / np.median(flux)
    win = win if win % 2 == 1 else win - 1
    order = min(poly_order, win - 1)
    if order % 2 == 0:
        order = max(order - 1, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from scipy.signal import savgol_filter

    trend = savgol_filter(flux, window_length=win, polyorder=order)
    trend = trend / np.median(trend)
    detrended = flux / trend
    detrended = detrended / np.median(detrended)
    logger.debug(
        "Savitzky-Golay detrend: window=%d points (%.2f d), order=%d",
        win, win * cadence_d, order,
    )
    return detrended


def normalize_target_id(target: str, mission: str = "TESS") -> str:
    """
    Normalize a target identifier for MAST queries.
    
    Parameters
    ----------
    target : str
        Target identifier (e.g., "307210830", "tic307210830", "TIC 307210830")
    mission : str
        Mission name ("TESS" or "Kepler")
        
    Returns
    -------
    str
        Normalized target ID (e.g., "TIC 307210830" or "KIC 11446443")
    """
    if target is None:
        return target
    
    # Strip Unicode BOM first, then whitespace
    cleaned = target.lstrip("\ufeff").strip()
    
    mission_upper = mission.upper()
    
    if mission_upper == "TESS":
        # Check if it's already in TIC format (case-insensitive)
        if cleaned.upper().startswith("TIC"):
            # Normalize spacing: "TIC307210830" or "tic 307210830" -> "TIC 307210830"
            parts = cleaned.split()
            if len(parts) == 1 and parts[0].upper().startswith("TIC"):
                # No space after TIC
                num_part = parts[0][3:]
                if num_part.isdigit():
                    return f"TIC {num_part}"
            elif len(parts) >= 2 and parts[0].upper() == "TIC":
                # Has space after TIC, e.g., "tic 307210830"
                num_part = parts[1]
                if num_part.isdigit():
                    return f"TIC {num_part}"
            return f"TIC {parts[-1]}" if parts[-1].isdigit() else cleaned
        # Bare numeric ID
        elif cleaned.isdigit():
            return f"TIC {cleaned}"
        # Lowercase tic prefix without space: "tic307210830"
        elif cleaned.lower().startswith("tic"):
            num_part = cleaned[3:].strip()
            if num_part.isdigit():
                return f"TIC {num_part}"
    
    elif mission_upper == "KEPLER":
        # Check if it's already in KIC format (case-insensitive)
        if cleaned.upper().startswith("KIC"):
            parts = cleaned.split()
            if len(parts) == 1 and parts[0].upper().startswith("KIC"):
                num_part = parts[0][3:]
                if num_part.isdigit():
                    return f"KIC {num_part}"
            elif len(parts) >= 2 and parts[0].upper() == "KIC":
                num_part = parts[1]
                if num_part.isdigit():
                    return f"KIC {num_part}"
            return f"KIC {parts[-1]}" if parts[-1].isdigit() else cleaned
        # Bare numeric ID
        elif cleaned.isdigit():
            return f"KIC {cleaned}"
        # Lowercase kic prefix without space: "kic11446443"
        elif cleaned.lower().startswith("kic"):
            num_part = cleaned[3:].strip()
            if num_part.isdigit():
                return f"KIC {num_part}"
    
    # Return as-is if no normalization rule applies
    return cleaned


def sanitize_filename(text: str) -> str:
    """
    Sanitize a string for use in filenames.
    
    Replaces spaces and special characters with underscores.
    
    Parameters
    ----------
    text : str
        Input string
        
    Returns
    -------
    str
        Sanitized string safe for filenames
    """
    # Replace spaces with underscores
    sanitized = text.replace(" ", "_")
    # Remove any characters that aren't ASCII alphanumeric, underscore, hyphen, or dot
    import re
    sanitized = re.sub(r'[^A-Za-z0-9_\-\.]', '', sanitized)
    return sanitized
