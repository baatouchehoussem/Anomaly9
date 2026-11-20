"""
Anomaly9 Persistent Candidate Storage
======================================
SQLite-backed database for candidate persistence with thread-safe connections.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from anomaly9.config import Anomaly9Config
from anomaly9.pipeline import LightCurveData
from anomaly9.vetting import VettingReport

logger = logging.getLogger("anomaly9.storage")


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a field from an object or dict.
    
    Parameters
    ----------
    obj : Any
        Object (dataclass, class instance) or dict.
    key : str
        Field name to retrieve.
    default : Any
        Default value if field not found.
    
    Returns
    -------
    Any
        Field value or default.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

SCHEMA_SQL = """
-- Run metadata table
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    config_hash TEXT NOT NULL,
    target_count INTEGER NOT NULL,
    notes TEXT
);

-- Candidates table with upsert on tic_id
CREATE TABLE IF NOT EXISTS candidates (
    tic_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    period_days REAL,
    t0_btjd REAL,
    depth_ppm REAL,
    duration_hours REAL,
    snr REAL,
    isolation_score REAL,
    vetting_score REAL,
    disposition TEXT NOT NULL,
    stellar_radius REAL,
    stellar_mass REAL,
    stellar_density REAL,
    ruwe REAL,
    exofop_toi TEXT,
    exofop_disp TEXT,
    checks_json TEXT,
    reviewed INTEGER DEFAULT 0,
    reviewer_notes TEXT,
    updated_at REAL NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_candidates_disposition ON candidates (disposition);
CREATE INDEX IF NOT EXISTS idx_candidates_vetting_score ON candidates (vetting_score);
CREATE INDEX IF NOT EXISTS idx_candidates_run_id ON candidates (run_id);
CREATE INDEX IF NOT EXISTS idx_candidates_reviewed ON candidates (reviewed);
"""


# --------------------------------------------------------------------------
# CandidateDB Class
# --------------------------------------------------------------------------

class CandidateDB:
    """
    Thread-safe SQLite database for candidate storage.
    
    Uses WAL mode for concurrent reads and single-writer safety.
    All connections are thread-local to avoid cross-thread issues.
    """
    
    def __init__(
        self,
        url: str = "sqlite:///candidates.db",
        wal_mode: bool = True,
        busy_timeout: int = 30000,
    ):
        """
        Initialize the database.
        
        Parameters
        ----------
        url : str
            SQLite URL. Use 'sqlite:///path.db' for file, 'sqlite:///:memory:' for memory.
        wal_mode : bool
            Enable WAL mode for better concurrency.
        busy_timeout : int
            Busy timeout in milliseconds.
        """
        # Parse URL
        if url.startswith("sqlite:///"):
            db_path = url[10:]  # Remove 'sqlite:///'
        elif url.startswith("sqlite://"):
            db_path = url[9:]
        else:
            raise ValueError(f"Invalid SQLite URL: {url}")
        
        self.db_path = db_path
        self.wal_mode = wal_mode
        self.busy_timeout = busy_timeout
        
        # Thread-local storage for connections
        self._local = threading.local()
        
        # Initialize schema
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            # Handle in-memory database specially
            if self.db_path == ":memory:":
                conn = sqlite3.connect(":memory:", check_same_thread=False)
            else:
                # Ensure directory exists
                db_dir = Path(self.db_path).parent
                if db_dir and not db_dir.exists():
                    db_dir.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Configure connection
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout}")
            
            if self.wal_mode and self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
            
            self._local.conn = conn
        
        return self._local.conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    
    def close(self) -> None:
        """Close thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
    
    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for explicit transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    # ----------------------------------------------------------------------
    # Run Management
    # ----------------------------------------------------------------------
    
    def create_run(
        self,
        run_id: str,
        config_hash: str,
        target_count: int,
        notes: str = "",
    ) -> None:
        """Record a new pipeline run."""
        conn = self._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, timestamp, config_hash, target_count, notes) VALUES (?, ?, ?, ?, ?)",
            (run_id, time.time(), config_hash, target_count, notes),
        )
        conn.commit()
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run metadata by ID."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None
    
    def list_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent runs."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    
    # ----------------------------------------------------------------------
    # Candidate CRUD
    # ----------------------------------------------------------------------
    
    def upsert_candidate(
        self,
        report: VettingReport,
        lc_data: Optional[LightCurveData] = None,
        run_id: str = "",
    ) -> None:
        """
        Insert or update a candidate.
        
        Parameters
        ----------
        report : VettingReport or dict
            Vetting report with detection and vetting results.
        lc_data : LightCurveData or dict, optional
            Light curve data for stellar metadata.
        run_id : str
            Associated run ID.
        """
        conn = self._get_connection()
        
        # Extract stellar metadata from lc_data if available
        stellar_radius = None
        stellar_mass = None
        stellar_density = None
        ruwe = None
        exofop_toi = None
        exofop_disp = None
        
        if lc_data:
            from anomaly9.catalog import (
                get_stellar_radius, get_stellar_density, get_ruwe,
                get_exofop_disposition
            )
            # Helper functions handle both dict and object
            stellar_radius = get_stellar_radius(lc_data)
            stellar_density = get_stellar_density(lc_data)
            ruwe = get_ruwe(lc_data)
            exofop_disp = get_exofop_disposition(lc_data)
            # TOI from metadata dict
            meta = _get_field(lc_data, "metadata", {})
            exofop_toi = meta.get("exofop_toi") if isinstance(meta, dict) else None
        
        # Serialize checks to JSON - handle both object and dict for check_details
        check_details = _get_field(report, "check_details", [])
        
        def _to_py(val):
            """Convert numpy types to Python native types for JSON serialization."""
            import numpy as np
            if isinstance(val, (np.bool_, np.bool)):
                return bool(val)
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            if isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val)
            return val
        
        checks_json = json.dumps([
            {
                "name": _to_py(_get_field(c, "name")),
                "passed": _to_py(_get_field(c, "passed")),
                "value": _to_py(_get_field(c, "value")),
                "threshold": _to_py(_get_field(c, "threshold")),
                "description": _to_py(_get_field(c, "description")),
            }
            for c in check_details
        ])
        
        # Safely get metrics dict
        metrics = _get_field(report, "metrics", {})
        
        now = time.time()
        
        conn.execute("""
            INSERT INTO candidates (
                tic_id, run_id, period_days, t0_btjd, depth_ppm, duration_hours,
                snr, isolation_score, vetting_score, disposition,
                stellar_radius, stellar_mass, stellar_density, ruwe,
                exofop_toi, exofop_disp, checks_json, reviewed, reviewer_notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tic_id) DO UPDATE SET
                run_id = excluded.run_id,
                period_days = excluded.period_days,
                t0_btjd = excluded.t0_btjd,
                depth_ppm = excluded.depth_ppm,
                duration_hours = excluded.duration_hours,
                snr = excluded.snr,
                isolation_score = excluded.isolation_score,
                vetting_score = excluded.vetting_score,
                disposition = excluded.disposition,
                stellar_radius = excluded.stellar_radius,
                stellar_mass = excluded.stellar_mass,
                stellar_density = excluded.stellar_density,
                ruwe = excluded.ruwe,
                exofop_toi = excluded.exofop_toi,
                exofop_disp = excluded.exofop_disp,
                checks_json = excluded.checks_json,
                updated_at = excluded.updated_at
        """, (
            _get_field(report, "tic_id"),
            run_id,
            metrics.get("period_d"),
            metrics.get("t0_bjd"),
            metrics.get("depth_ppm"),
            metrics.get("duration_d", 0) * 24.0 if metrics.get("duration_d") else None,
            metrics.get("snr"),
            metrics.get("isolation_score"),
            _get_field(report, "confidence_score"),
            _get_field(report, "disposition"),
            stellar_radius,
            stellar_mass,
            stellar_density,
            ruwe,
            exofop_toi,
            exofop_disp,
            checks_json,
            0,  # reviewed (default false)
            "",  # reviewer_notes (default empty)
            now,
        ))
        conn.commit()
    
    def get_by_tic(self, tic_id: str) -> Optional[Dict[str, Any]]:
        """Get candidate by TIC ID."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM candidates WHERE tic_id = ?", (tic_id,)).fetchone()
        if row:
            result = dict(row)
            # Parse checks_json
            if result.get("checks_json"):
                try:
                    result["checks"] = json.loads(result["checks_json"])
                except json.JSONDecodeError:
                    result["checks"] = []
            else:
                result["checks"] = []
            return result
        return None
    
    def query_candidates(
        self,
        disposition: Optional[str] = None,
        min_score: Optional[float] = None,
        reviewed: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query candidates with filters.
        
        Parameters
        ----------
        disposition : str, optional
            Filter by disposition (CANDIDATE, AMBIGUOUS, FALSE_POSITIVE, KNOWN_FP).
        min_score : float, optional
            Minimum vetting score.
        reviewed : bool, optional
            Filter by reviewed status.
        limit : int
            Maximum results.
        offset : int
            Pagination offset.
        
        Returns
        -------
        List[dict]
            Matching candidates.
        """
        conn = self._get_connection()
        
        conditions = []
        params = []
        
        if disposition:
            conditions.append("disposition = ?")
            params.append(disposition)
        
        if min_score is not None:
            conditions.append("vetting_score >= ?")
            params.append(min_score)
        
        if reviewed is not None:
            conditions.append("reviewed = ?")
            params.append(1 if reviewed else 0)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT * FROM candidates
            {where_clause}
            ORDER BY vetting_score DESC, updated_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        rows = conn.execute(query, params).fetchall()
        
        results = []
        for row in rows:
            result = dict(row)
            if result.get("checks_json"):
                try:
                    result["checks"] = json.loads(result["checks_json"])
                except json.JSONDecodeError:
                    result["checks"] = []
            else:
                result["checks"] = []
            results.append(result)
        
        return results
    
    def mark_reviewed(self, tic_id: str, notes: str = "") -> bool:
        """Mark a candidate as reviewed with optional notes."""
        conn = self._get_connection()
        cursor = conn.execute(
            "UPDATE candidates SET reviewed = 1, reviewer_notes = ?, updated_at = ? WHERE tic_id = ?",
            (notes, time.time(), tic_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    
    def count_candidates(self, disposition: Optional[str] = None) -> int:
        """Count candidates, optionally filtered by disposition."""
        conn = self._get_connection()
        if disposition:
            row = conn.execute(
                "SELECT COUNT(*) FROM candidates WHERE disposition = ?", (disposition,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()
        return row[0] if row else 0
    
    # ----------------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------------
    
    def export_to_csv_json(
        self,
        output_dir: Union[str, Path],
        disposition: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Export candidates to CSV and JSON (backward compatible with candidates_ranked.*).
        
        Parameters
        ----------
        output_dir : Path
            Output directory.
        disposition : str, optional
            Filter by disposition.
        
        Returns
        -------
        Tuple[Path, Path]
            Paths to (csv_file, json_file).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        candidates = self.query_candidates(disposition=disposition, limit=10000)
        
        # CSV
        csv_path = output_dir / "candidates_ranked.csv"
        import csv
        if candidates:
            fieldnames = [
                "tic_id", "run_id", "period_days", "t0_btjd", "depth_ppm",
                "duration_hours", "snr", "isolation_score", "vetting_score",
                "disposition", "stellar_radius", "stellar_mass", "stellar_density",
                "ruwe", "exofop_toi", "exofop_disp", "reviewed", "reviewer_notes",
                "updated_at"
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for c in candidates:
                    row = {k: c.get(k) for k in fieldnames}
                    writer.writerow(row)
        
        # JSON
        json_path = output_dir / "candidates_ranked.json"
        with open(json_path, "w") as f:
            json.dump(candidates, f, indent=2)
        
        logger.info("Exported %d candidates to %s and %s", len(candidates), csv_path, json_path)
        return csv_path, json_path
    
    # ----------------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------------
    
    def vacuum(self) -> None:
        """Vacuum database to reclaim space."""
        conn = self._get_connection()
        conn.execute("VACUUM")
    
    def __enter__(self) -> "CandidateDB":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# --------------------------------------------------------------------------
# Module Exports
# --------------------------------------------------------------------------

__all__ = [
    "CandidateDB",
]