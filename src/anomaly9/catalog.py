"""
Anomaly9 Catalog Cross-Match Engine
====================================
Queries TIC, Gaia DR3, and ExoFOP-TESS for stellar metadata and known dispositions.
Implements local caching with TTL to avoid repeated network calls.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

# Optional imports for catalog queries (may not be available)
try:
    from astroquery.vizier import Vizier
except ImportError:
    Vizier = None

try:
    from astroquery.gaia import Gaia
except ImportError:
    Gaia = None

try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except ImportError:
    SkyCoord = None
    u = None

logger = logging.getLogger("anomaly9.catalog")

# Cache configuration
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "anomaly9" / "catalog"
DEFAULT_TTL_DAYS = 30
DEFAULT_TIMEOUT_SECONDS = 10.0

# Solar density in g/cm3
SOLAR_DENSITY_GCC = 1.408

@dataclass
class StellarMetadata:
    """Stellar metadata from TIC and/or Gaia DR3."""
    tic_id: str
    gaia_dr3_id: Optional[int] = None
    radius_solar: Optional[float] = None
    mass_solar: Optional[float] = None
    teff_k: Optional[float] = None
    logg: Optional[float] = None
    ruwe: Optional[float] = None
    stellar_density_gcc: Optional[float] = None
    source: str = "TIC"

    def __post_init__(self):
        if self.mass_solar is not None and self.radius_solar is not None and self.radius_solar > 0:
            self.stellar_density_gcc = (self.mass_solar / (self.radius_solar ** 3)) * SOLAR_DENSITY_GCC

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StellarMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class ExofopDisposition:
    """ExoFOP-TESS disposition for a TIC."""
    tic_id: str
    toi: Optional[str] = None
    tfopwg_disp: Optional[str] = None
    is_known_fp: bool = False
    is_known_planet: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExofopDisposition":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

class CatalogCache:
    """File-based cache with TTL for catalog queries."""
    def __init__(self, cache_dir: Optional[Path] = None, ttl_days: int = DEFAULT_TTL_DAYS):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_days * 86400

    def _get_cache_path(self, source: str, tic_id: str) -> Path:
        safe_id = tic_id.replace(" ", "_").replace("/", "_")
        return self.cache_dir / f"{source}_{safe_id}.pkl"

    def get(self, source: str, tic_id: str) -> Optional[Any]:
        """Retrieve cached data if not expired."""
        cache_path = self._get_cache_path(source, tic_id)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if time.time() - cached["timestamp"] > self.ttl_seconds:
                logger.debug("Cache expired for %s %s", source, tic_id)
                return None
            logger.debug("Cache hit for %s %s", source, tic_id)
            return cached["data"]
        except Exception as e:
            logger.warning("Failed to read cache for %s %s: %s", source, tic_id, e)
            return None

    def set(self, source: str, tic_id: str, data: Any) -> None:
        """Store data in cache."""
        cache_path = self._get_cache_path(source, tic_id)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"timestamp": time.time(), "data": data}, f)
            logger.debug("Cached %s %s", source, tic_id)
        except Exception as e:
            logger.warning("Failed to write cache for %s %s: %s", source, tic_id, e)

# Global cache instance
_cache = CatalogCache()

def _normalize_tic_id(tic_id: str) -> str:
    """Normalize TIC ID to numeric string."""
    if tic_id.upper().startswith("TIC"):
        return tic_id.split()[-1]
    return tic_id

def query_tic_v9(tic_id: str, cache: Optional[CatalogCache] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Optional[StellarMetadata]:
    """Query TIC v9 catalog via Vizier for stellar parameters.
    Parameters
    ----------
    tic_id : str
        TIC identifier (e.g., "25155310" or "TIC 25155310")
    cache : CatalogCache, optional
        Cache instance to use (default: global)
    timeout : float
        Network timeout in seconds
    Returns
    -------
    StellarMetadata or None"""
    if Vizier is None:
        logger.warning("astroquery.vizier not available for TIC query")
        return None
    
    cache = cache or _cache
    norm_id = _normalize_tic_id(tic_id)
    cached = cache.get("tic", norm_id)
    if cached is not None:
        return cached
    try:
        Vizier.ROW_LIMIT = 1
        Vizier.TIMEOUT = timeout
        result = Vizier.query_constraints(catalog="I/359/tic", TIC=norm_id)
        if not result or len(result) == 0:
            logger.debug("No TIC v9 entry for %s", norm_id)
            return None
        table = result[0]
        if len(table) == 0:
            return None
        row = table[0]
        metadata = StellarMetadata(tic_id=f"TIC {norm_id}", source="TIC")
        if "Teff" in table.colnames and row["Teff"] is not np.ma.masked:
            metadata.teff_k = float(row["Teff"])
        if "logg" in table.colnames and row["logg"] is not np.ma.masked:
            metadata.logg = float(row["logg"])
        if "rad" in table.colnames and row["rad"] is not np.ma.masked:
            metadata.radius_solar = float(row["rad"])
        if "mass" in table.colnames and row["mass"] is not np.ma.masked:
            metadata.mass_solar = float(row["mass"])
        if metadata.mass_solar is not None and metadata.radius_solar is not None:
            metadata.stellar_density_gcc = (metadata.mass_solar / (metadata.radius_solar ** 3)) * SOLAR_DENSITY_GCC
        logger.info("TIC v9 query successful for TIC %s", norm_id)
        cache.set("tic", norm_id, metadata)
        return metadata
    except Exception as e:
        logger.warning("TIC v9 query failed for %s: %s", norm_id, e)
        return None

def query_gaia_dr3(tic_id: str, cache: Optional[CatalogCache] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Optional[StellarMetadata]:
    """Query Gaia DR3 via astroquery for stellar parameters.
    Uses TIC-to-Gaia cross-match via Vizier first, then queries Gaia.
    Parameters
    ----------
    tic_id : str
        TIC identifier
    cache : CatalogCache, optional
        Cache instance to use
    timeout : float
        Network timeout in seconds
    Returns
    -------
    StellarMetadata or None"""
    if Vizier is None or Gaia is None:
        logger.warning("astroquery not available for Gaia query")
        return None
    
    cache = cache or _cache
    norm_id = _normalize_tic_id(tic_id)
    cached = cache.get("gaia", norm_id)
    if cached is not None:
        return cached
    gaia_id = None
    try:
        Vizier.ROW_LIMIT = 1
        Vizier.TIMEOUT = timeout
        result = Vizier.query_constraints(catalog="I/359/tic", TIC=norm_id)
        if result and len(result) > 0:
            table = result[0]
            if len(table) > 0 and "GAIA" in table.colnames:
                gaia_val = table[0]["GAIA"]
                if gaia_val is not np.ma.masked:
                    gaia_id = int(gaia_val)
        if gaia_id is None:
            logger.debug("No Gaia DR3 cross-match for TIC %s", norm_id)
            return None
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        Gaia.TIMEOUT = timeout
        query = f"""SELECT source_id, radius_gspphot, mass_gspphot, teff_gspphot, logg_gspphot, ruwe FROM gaiadr3.gaia_source WHERE source_id = {gaia_id}"""
        job = Gaia.launch_job_async(query)
        result_table = job.get_results()
        if len(result_table) == 0:
            logger.debug("No Gaia DR3 entry for source_id %s", gaia_id)
            return None
        row = result_table[0]
        metadata = StellarMetadata(tic_id=f"TIC {norm_id}", gaia_dr3_id=gaia_id, source="Gaia")
        if "radius_gspphot" in result_table.colnames and row["radius_gspphot"] is not np.ma.masked:
            metadata.radius_solar = float(row["radius_gspphot"])
        if "mass_gspphot" in result_table.colnames and row["mass_gspphot"] is not np.ma.masked:
            metadata.mass_solar = float(row["mass_gspphot"])
        if "teff_gspphot" in result_table.colnames and row["teff_gspphot"] is not np.ma.masked:
            metadata.teff_k = float(row["teff_gspphot"])
        if "logg_gspphot" in result_table.colnames and row["logg_gspphot"] is not np.ma.masked:
            metadata.logg = float(row["logg_gspphot"])
        if "ruwe" in result_table.colnames and row["ruwe"] is not np.ma.masked:
            metadata.ruwe = float(row["ruwe"])
        if metadata.mass_solar is not None and metadata.radius_solar is not None:
            metadata.stellar_density_gcc = (metadata.mass_solar / (metadata.radius_solar ** 3)) * SOLAR_DENSITY_GCC
        logger.info("Gaia DR3 query successful for TIC %s (Gaia %s)", norm_id, gaia_id)
        cache.set("gaia", norm_id, metadata)
        return metadata
    except Exception as e:
        logger.warning("Gaia DR3 query failed for %s: %s", norm_id, e)
        return None


def query_gaia_dr3_by_coords(ra: float, dec: float, radius_arcsec: float = 2.0, cache: Optional[CatalogCache] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Optional[StellarMetadata]:
    """Query Gaia DR3 by sky coordinates (cone search).
    Parameters
    ----------
    ra : float
        Right Ascension [degrees]
    dec : float
        Declination [degrees]
    radius_arcsec : float
        Search radius [arcseconds]
    cache : CatalogCache, optional
        Cache instance to use
    timeout : float
        Network timeout in seconds
    Returns
    -------
    StellarMetadata or None"""
    if Gaia is None or SkyCoord is None or u is None:
        logger.warning("astroquery or astropy not available for Gaia cone search")
        return None
    
    cache = cache or _cache
    cache_key = f"cone_{ra:.6f}_{dec:.6f}_{radius_arcsec}"
    cached = cache.get("gaia", cache_key)
    if cached is not None:
        return cached
    try:
        Gaia.ROW_LIMIT = 1
        Gaia.TIMEOUT = timeout
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        job = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec)
        result = job.get_results()
        if len(result) > 0:
            row = result[0]
            gaia_id = int(row["source_id"]) if row["source_id"] is not np.ma.masked else None
            ruwe = float(row["ruwe"]) if "ruwe" in result.colnames and row["ruwe"] is not np.ma.masked else None
            metadata = StellarMetadata(tic_id="", gaia_dr3_id=gaia_id, ruwe=ruwe, source="Gaia")
            cache.set("gaia", cache_key, metadata)
            logger.info("Gaia DR3 cone search successful at RA=%.6f, Dec=%.6f", ra, dec)
            return metadata
        return None
    except Exception as e:
        logger.warning("Gaia DR3 cone search failed: %s", e)
        return None


def query_exofop_tess(tic_id: str, cache: Optional[CatalogCache] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Optional[ExofopDisposition]:
    """Query ExoFOP-TESS for known TOI and disposition.
    Parameters
    ----------
    tic_id : str
        TIC identifier
    cache : CatalogCache, optional
        Cache instance to use
    timeout : float
        Network timeout in seconds
    Returns
    -------
    ExofopDisposition or None"""
    cache = cache or _cache
    norm_id = _normalize_tic_id(tic_id)
    cached = cache.get("exofop", norm_id)
    if cached is not None:
        return cached
    try:
        url = "https://exofop.ipac.caltech.edu/tess/download_toi.php"
        params = {"sort": "toi", "output": "json", "tic": norm_id}
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not data or len(data) == 0:
            logger.debug("No ExoFOP entry for TIC %s", norm_id)
            return None
        row = data[0]
        tfopwg_disp = row.get("tfopwg_disp", "")
        is_known_fp = tfopwg_disp in ("FP", "EB", "FA", "BLEND", "CP")
        is_known_planet = tfopwg_disp in ("CP", "KP", "PC")
        disposition = ExofopDisposition(tic_id=f"TIC {norm_id}", toi=str(row.get("toi", "")) if row.get("toi") else None, tfopwg_disp=tfopwg_disp if tfopwg_disp else None, is_known_fp=is_known_fp, is_known_planet=is_known_planet, notes=row.get("comments", "") if row.get("comments") else "")
        logger.info("ExoFOP query successful for TIC %s: TOI=%s, disp=%s", norm_id, disposition.toi, disposition.tfopwg_disp)
        cache.set("exofop", norm_id, disposition)
        return disposition
    except Exception as e:
        logger.warning("ExoFOP query failed for %s: %s", norm_id, e)
        return None

def merge_stellar_metadata(tic_meta: Optional[StellarMetadata], gaia_meta: Optional[StellarMetadata]) -> Optional[StellarMetadata]:
    """Merge TIC and Gaia metadata, preferring Gaia for physical parameters."""
    if tic_meta is None and gaia_meta is None:
        return None
    if tic_meta is None:
        gaia_meta.source = "Gaia"
        return gaia_meta
    if gaia_meta is None:
        tic_meta.source = "TIC"
        return tic_meta
    merged = StellarMetadata(tic_id=tic_meta.tic_id, gaia_dr3_id=gaia_meta.gaia_dr3_id, radius_solar=gaia_meta.radius_solar or tic_meta.radius_solar, mass_solar=gaia_meta.mass_solar or tic_meta.mass_solar, teff_k=gaia_meta.teff_k or tic_meta.teff_k, logg=gaia_meta.logg or tic_meta.logg, ruwe=gaia_meta.ruwe, source="TIC+Gaia")
    if merged.mass_solar is not None and merged.radius_solar is not None and merged.radius_solar > 0:
        merged.stellar_density_gcc = (merged.mass_solar / (merged.radius_solar ** 3)) * SOLAR_DENSITY_GCC
    elif tic_meta.stellar_density_gcc is not None:
        merged.stellar_density_gcc = tic_meta.stellar_density_gcc
    return merged

def enrich_with_catalogs(lc_data, enable_gaia: bool = True, enable_tic: bool = True, enable_exofop: bool = True, cache_dir: Optional[Path] = None, ttl_days: int = DEFAULT_TTL_DAYS, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
    """Enrich LightCurveData metadata with catalog cross-match results.
    Modifies lc_data.metadata in-place with:
    - stellar_metadata: StellarMetadata object
    - exofop_disposition: ExofopDisposition object
    Parameters
    ----------
    lc_data : LightCurveData
        Light curve data container
    enable_gaia : bool
        Query Gaia DR3
    enable_tic : bool
        Query TIC v9
    enable_exofop : bool
        Query ExoFOP-TESS
    cache_dir : Path, optional
        Custom cache directory
    ttl_days : int
        Cache TTL in days
    timeout : float
        Network timeout in seconds"""
    target = lc_data.target
    if not target.upper().startswith("TIC"):
        logger.warning("Target %s is not a TIC ID, skipping catalog enrichment", target)
        return
    tic_id = _normalize_tic_id(target)
    cache = CatalogCache(cache_dir, ttl_days) if cache_dir else _cache
    tic_meta = None
    if enable_tic:
        tic_meta = query_tic_v9(tic_id, cache=cache, timeout=timeout)
    gaia_meta = None
    if enable_gaia:
        gaia_meta = query_gaia_dr3(tic_id, cache=cache, timeout=timeout)
    merged_meta = merge_stellar_metadata(tic_meta, gaia_meta)
    exofop_disp = None
    if enable_exofop:
        exofop_disp = query_exofop_tess(tic_id, cache=cache, timeout=timeout)
    if merged_meta:
        lc_data.metadata["stellar_metadata"] = merged_meta
        lc_data.metadata["stellar_radius_rsun"] = merged_meta.radius_solar
        lc_data.metadata["stellar_mass_msun"] = merged_meta.mass_solar
        lc_data.metadata["stellar_teff_k"] = merged_meta.teff_k
        lc_data.metadata["stellar_logg"] = merged_meta.logg
        lc_data.metadata["stellar_density_gcc"] = merged_meta.stellar_density_gcc
        lc_data.metadata["gaia_dr3_id"] = merged_meta.gaia_dr3_id
        lc_data.metadata["catalog_source"] = merged_meta.source
    if exofop_disp:
        lc_data.metadata["exofop_disposition"] = exofop_disp
        lc_data.metadata["exofop_toi"] = exofop_disp.toi
        lc_data.metadata["exofop_tfopwg_disp"] = exofop_disp.tfopwg_disp
        lc_data.metadata["is_known_fp"] = exofop_disp.is_known_fp
        lc_data.metadata["is_known_planet"] = exofop_disp.is_known_planet


# --------------------------------------------------------------------------
# Helper functions for accessing catalog metadata
# --------------------------------------------------------------------------

def get_stellar_density(lc_data) -> Optional[float]:
    """Get stellar density [g/cm³] from LightCurveData metadata."""
    meta = lc_data.metadata.get("stellar_metadata")
    if meta is None:
        return None
    if isinstance(meta, StellarMetadata):
        return meta.stellar_density_gcc
    if isinstance(meta, dict):
        return meta.get("stellar_density_gcc")
    return None


def get_stellar_radius(lc_data) -> Optional[float]:
    """Get stellar radius [R_sun] from metadata."""
    meta = lc_data.metadata.get("stellar_metadata")
    if meta is None:
        return None
    if isinstance(meta, StellarMetadata):
        return meta.radius_solar
    if isinstance(meta, dict):
        return meta.get("stellar_radius_rsun") or meta.get("radius_solar")
    return None


def get_ruwe(lc_data) -> Optional[float]:
    """Get Gaia RUWE from metadata."""
    meta = lc_data.metadata.get("stellar_metadata")
    if meta is None:
        return None
    if isinstance(meta, StellarMetadata):
        return meta.ruwe
    if isinstance(meta, dict):
        return meta.get("ruwe")
    return None


def is_known_fp(lc_data) -> bool:
    """Check if target is a known false positive per ExoFOP."""
    disp = lc_data.metadata.get("exofop_disposition")
    if disp is None:
        return False
    if isinstance(disp, ExofopDisposition):
        return disp.is_known_fp
    if isinstance(disp, dict):
        return disp.get("is_known_fp", False)
    return False


def is_known_planet(lc_data) -> bool:
    """Check if target is a known planet per ExoFOP."""
    disp = lc_data.metadata.get("exofop_disposition")
    if disp is None:
        return False
    if isinstance(disp, ExofopDisposition):
        return disp.is_known_planet
    if isinstance(disp, dict):
        return disp.get("is_known_planet", False)
    return False


def get_exofop_disposition(lc_data) -> Optional[str]:
    """Get TFOPWG disposition string from ExoFOP."""
    disp = lc_data.metadata.get("exofop_disposition")
    if disp is None:
        return None
    if isinstance(disp, ExofopDisposition):
        return disp.tfopwg_disp
    if isinstance(disp, dict):
        return disp.get("tfopwg_disp") or disp.get("exofop_tfopwg_disp")
    return None


__all__ = [
    "StellarMetadata",
    "ExofopDisposition",
    "CatalogCache",
    "query_tic_v9",
    "query_gaia_dr3",
    "query_exofop_tess",
    "merge_stellar_metadata",
    "enrich_with_catalogs",
    "get_stellar_density",
    "get_stellar_radius",
    "get_ruwe",
    "is_known_fp",
    "is_known_planet",
    "get_exofop_disposition",
    "SOLAR_DENSITY_GCC",
]
