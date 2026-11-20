"""
Anomaly9 Configuration System
=============================
Structured configuration with TOML file support, CLI overrides, and defaults.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import tomllib
except ImportError:
    import tomli as tomllib

logger = logging.getLogger("anomaly9.config")


# --------------------------------------------------------------------------
# Configuration Dataclasses
# --------------------------------------------------------------------------

@dataclass
class MastConfig:
    """MAST ingestion configuration."""
    rate_limit: float = 1.0
    cloud_dataset: bool = True
    timeout: int = 300
    max_sectors: int = 6


@dataclass
class DetectorConfig:
    """Dual detector configuration."""
    period_min: float = 0.5
    period_max: float = 10.0
    period_step: float = 0.002
    contamination: float = 0.01
    z_window: int = 25
    bls_top_n: int = 5


@dataclass
class VettingConfig:
    """Astrophysical vetting thresholds."""
    snr_threshold: float = 7.0
    odd_even_threshold: float = 0.3
    spnr_threshold: float = 7.0
    duration_min_ratio: float = 0.001
    duration_max_ratio: float = 0.1
    secondary_eclipse_threshold: float = 0.1
    transit_shape_threshold: float = 0.7
    momentum_dump_tolerance: float = 0.2


@dataclass
class CatalogConfig:
    """External catalog cross-match configuration."""
    enable_gaia: bool = True
    enable_tic: bool = True
    enable_exofop: bool = True
    cache_ttl_days: int = 30
    cache_dir: Optional[str] = None
    timeout_seconds: float = 10.0


@dataclass
class OutputConfig:
    """Output and diagnostics configuration."""
    output_dir: str = "output"
    diagnostics_dir: str = "diagnostics"
    cache_dir: str = "data/cache"
    plot_dpi: int = 300
    plot_format: str = "png"


@dataclass
class DatabaseConfig:
    """SQLite database configuration."""
    url: str = "sqlite:///candidates.db"
    wal_mode: bool = True
    busy_timeout: int = 30000


@dataclass
class Anomaly9Config:
    """Top-level Anomaly9 configuration."""
    mast: MastConfig = field(default_factory=MastConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    vetting: VettingConfig = field(default_factory=VettingConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Internal: computed config hash for run tracking
    _config_hash: str = field(default="", repr=False)

    def __post_init__(self):
        # Hash will be computed on demand via property
        pass

    def compute_hash(self) -> str:
        """Compute a stable hash of the configuration (excluding runtime paths)."""
        # Create a serializable dict excluding paths that vary by machine
        config_dict = {
            "mast": asdict(self.mast),
            "detector": asdict(self.detector),
            "vetting": asdict(self.vetting),
            "catalog": {k: v for k, v in asdict(self.catalog).items() if k != "cache_dir"},
            "output": {k: v for k, v in asdict(self.output).items() if k not in ("output_dir", "diagnostics_dir", "cache_dir")},
            "database": {k: v for k, v in asdict(self.database).items() if k != "url"},
        }
        serialized = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        """Return the configuration hash (computed on demand)."""
        return self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mast": asdict(self.mast),
            "detector": asdict(self.detector),
            "vetting": asdict(self.vetting),
            "catalog": asdict(self.catalog),
            "output": asdict(self.output),
            "database": asdict(self.database),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Anomaly9Config":
        """Create config from dictionary (e.g., parsed TOML)."""
        return cls(
            mast=MastConfig(**data.get("mast", {})),
            detector=DetectorConfig(**data.get("detector", {})),
            vetting=VettingConfig(**data.get("vetting", {})),
            catalog=CatalogConfig(**data.get("catalog", {})),
            output=OutputConfig(**data.get("output", {})),
            database=DatabaseConfig(**data.get("database", {})),
        )


# --------------------------------------------------------------------------
# Configuration Loading
# --------------------------------------------------------------------------

DEFAULT_CONFIG_PATHS = [
    Path("anomaly9.toml"),
    Path("config/anomaly9.toml"),
    Path.home() / ".config" / "anomaly9" / "anomaly9.toml",
]


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Anomaly9Config:
    """
    Load configuration from TOML file with CLI overrides.
    
    Priority (highest to lowest):
    1. CLI overrides (cli_overrides dict)
    2. TOML file (config_path or discovered default)
    3. Built-in defaults
    
    Parameters
    ----------
    config_path : Path, optional
        Explicit path to TOML config file. If None, searches default locations.
    cli_overrides : dict, optional
        Flattened CLI arguments to override config (e.g., {"detector.period_min": 1.0}).
    
    Returns
    -------
    Anomaly9Config
        Fully resolved configuration.
    """
    # Start with defaults
    config_data = {}
    
    # Load TOML file if found
    toml_path = None
    if config_path:
        toml_path = Path(config_path)
        if not toml_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        for path in DEFAULT_CONFIG_PATHS:
            if path.exists():
                toml_path = path
                break
    
    if toml_path:
        logger.info("Loading configuration from %s", toml_path)
        with open(toml_path, "rb") as f:
            config_data = tomllib.load(f)
    else:
        logger.info("No config file found; using defaults")
    
    # Apply CLI overrides (flattened dict with dot notation)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is None:
                continue
            parts = key.split(".")
            if len(parts) != 2:
                logger.warning("Invalid override key format: %s (expected section.key)", key)
                continue
            section, param = parts
            if section not in config_data:
                config_data[section] = {}
            config_data[section][param] = value
    
    return Anomaly9Config.from_dict(config_data)


def generate_default_toml(output_path: Union[str, Path]) -> None:
    """
    Generate a default anomaly9.toml configuration file.
    
    Parameters
    ----------
    output_path : Path
        Where to write the default config file.
    """
    config = Anomaly9Config()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate TOML with comments
    toml_content = """# Anomaly9 Configuration
# ======================
# Generated by anomaly9.config.generate_default_toml()

[mast]
# Rate limit between MAST queries (seconds)
rate_limit = 1.0
# Use AWS S3 cloud dataset streaming for faster downloads
cloud_dataset = true
# Request timeout in seconds
timeout = 300
# Maximum sectors to process per target
max_sectors = 6

[detector]
# Period search range [days]
period_min = 0.5
period_max = 10.0
# Period grid step [days]
period_step = 0.002
# Isolation Forest contamination rate
contamination = 0.01
# Rolling Z-score window size
z_window = 25
# Number of top BLS peaks to retain
bls_top_n = 5

[vetting]
# SNR threshold for candidate acceptance
snr_threshold = 7.0
# Odd/even transit depth fractional difference threshold
odd_even_threshold = 0.3
# Signal-to-Pink-Noise Ratio threshold
spnr_threshold = 7.0
# Duration/period ratio limits (heuristic fallback)
duration_min_ratio = 0.001
duration_max_ratio = 0.1
# Secondary eclipse depth ratio threshold
secondary_eclipse_threshold = 0.1
# Transit shape boxiness threshold (box/trapezoid RMS)
transit_shape_threshold = 0.7
# Momentum dump proximity tolerance [days]
momentum_dump_tolerance = 0.2

[catalog]
# Enable Gaia DR3 cross-match
enable_gaia = true
# Enable TIC v9 catalog query
enable_tic = true
# Enable ExoFOP-TESS disposition lookup
enable_exofop = true
# Cache TTL in days
cache_ttl_days = 30
# Custom cache directory (optional)
# cache_dir = "~/.cache/anomaly9/catalog"
# Network timeout in seconds
timeout_seconds = 10.0

[output]
# Output directory for results
output_dir = "output"
# Diagnostics plots directory
diagnostics_dir = "diagnostics"
# FITS cache directory
cache_dir = "data/cache"
# Plot DPI
plot_dpi = 300
# Plot format (png, pdf, svg)
plot_format = "png"

[database]
# SQLite database URL (use sqlite:///path.db for file, sqlite:///:memory: for in-memory)
url = "sqlite:///candidates.db"
# Enable WAL mode for better concurrency
wal_mode = true
# Busy timeout in milliseconds
busy_timeout = 30000
"""
    
    with open(output_path, "w") as f:
        f.write(toml_content)
    
    logger.info("Generated default config at %s", output_path)


# --------------------------------------------------------------------------
# Convenience: Configuration for Pipeline Functions
# --------------------------------------------------------------------------

def config_to_pipeline_kwargs(config: Anomaly9Config) -> Dict[str, Any]:
    """Extract pipeline-relevant parameters from config."""
    return {
        "period_min": config.detector.period_min,
        "period_max": config.detector.period_max,
        "period_step": config.detector.period_step,
        "contamination": config.detector.contamination,
        "z_window": config.detector.z_window,
        "bls_top_n": config.detector.bls_top_n,
        "rate_limit": config.mast.rate_limit,
        "max_sectors": config.mast.max_sectors,
        "enable_gaia": config.catalog.enable_gaia,
        "enable_tic": config.catalog.enable_tic,
        "enable_exofop": config.catalog.enable_exofop,
    }


def config_to_vetting_kwargs(config: Anomaly9Config) -> Dict[str, Any]:
    """Extract vetting-relevant parameters from config."""
    return {
        "snr_threshold": config.vetting.snr_threshold,
        "odd_even_threshold": config.vetting.odd_even_threshold,
        "spnr_threshold": config.vetting.spnr_threshold,
        "duration_min_ratio": config.vetting.duration_min_ratio,
        "duration_max_ratio": config.vetting.duration_max_ratio,
        "secondary_eclipse_threshold": config.vetting.secondary_eclipse_threshold,
        "transit_shape_threshold": config.vetting.transit_shape_threshold,
        "momentum_dump_tolerance": config.vetting.momentum_dump_tolerance,
    }


# --------------------------------------------------------------------------
# Module Exports
# --------------------------------------------------------------------------

__all__ = [
    "MastConfig",
    "DetectorConfig",
    "VettingConfig",
    "CatalogConfig",
    "OutputConfig",
    "DatabaseConfig",
    "Anomaly9Config",
    "load_config",
    "generate_default_toml",
    "config_to_pipeline_kwargs",
    "config_to_vetting_kwargs",
]