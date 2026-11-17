"""
Unit tests for Anomaly9 CLI normalization and catalog parsing.
"""

import tempfile
from pathlib import Path

import pytest

from anomaly9.pipeline import normalize_target_id, sanitize_filename
from anomaly9.cli import parse_catalog


class TestNormalizeTargetId:
    """Tests for normalize_target_id function."""

    # TESS tests
    def test_tess_bare_numeric(self):
        """Bare numeric TESS ID gets 'TIC ' prefix."""
        assert normalize_target_id("307210830", "TESS") == "TIC 307210830"
    
    def test_tess_lowercase_tic_compact(self):
        """Lowercase 'tic' prefix without space gets normalized."""
        assert normalize_target_id("tic307210830", "TESS") == "TIC 307210830"
    
    def test_tess_lowercase_tic_with_space(self):
        """Lowercase 'tic' with space gets normalized."""
        assert normalize_target_id("tic 307210830", "TESS") == "TIC 307210830"
    
    def test_tess_uppercase_tic_compact(self):
        """Uppercase 'TIC' without space gets normalized."""
        assert normalize_target_id("TIC307210830", "TESS") == "TIC 307210830"
    
    def test_tess_already_normalized(self):
        """Already normalized 'TIC 307210830' stays the same."""
        assert normalize_target_id("TIC 307210830", "TESS") == "TIC 307210830"
    
    def test_tess_strips_whitespace(self):
        """Whitespace around ID is stripped."""
        assert normalize_target_id("  307210830  ", "TESS") == "TIC 307210830"
    
    def test_tess_strips_bom(self):
        """Unicode BOM is stripped."""
        assert normalize_target_id("\ufeff307210830", "TESS") == "TIC 307210830"
    
    def test_tess_bom_and_whitespace(self):
        """BOM and whitespace both stripped."""
        assert normalize_target_id("\ufeff  307210830  ", "TESS") == "TIC 307210830"
    
    # Kepler tests
    def test_kepler_bare_numeric(self):
        """Bare numeric Kepler ID gets 'KIC ' prefix."""
        assert normalize_target_id("11446443", "KEPLER") == "KIC 11446443"
    
    def test_kepler_lowercase_kic_compact(self):
        """Lowercase 'kic' prefix without space gets normalized."""
        assert normalize_target_id("kic11446443", "KEPLER") == "KIC 11446443"
    
    def test_kepler_lowercase_kic_with_space(self):
        """Lowercase 'kic' with space gets normalized."""
        assert normalize_target_id("kic 11446443", "KEPLER") == "KIC 11446443"
    
    def test_kepler_uppercase_kic_compact(self):
        """Uppercase 'KIC' without space gets normalized."""
        assert normalize_target_id("KIC11446443", "KEPLER") == "KIC 11446443"
    
    def test_kepler_already_normalized(self):
        """Already normalized 'KIC 11446443' stays the same."""
        assert normalize_target_id("KIC 11446443", "KEPLER") == "KIC 11446443"
    
    def test_kepler_strips_bom(self):
        """Unicode BOM is stripped for Kepler."""
        assert normalize_target_id("\ufeff11446443", "KEPLER") == "KIC 11446443"
    
    # Edge cases
    def test_unknown_mission_passthrough(self):
        """Unknown mission passes through unchanged (after stripping)."""
        assert normalize_target_id("  12345  ", "UNKNOWN") == "12345"
    
    def test_non_numeric_passthrough(self):
        """Non-numeric IDs pass through (after stripping)."""
        assert normalize_target_id("  HD 209458  ", "TESS") == "HD 209458"
    
    def test_none_input(self):
        """None input returns None."""
        assert normalize_target_id(None, "TESS") is None


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_spaces_to_underscores(self):
        """Spaces become underscores."""
        assert sanitize_filename("TIC 307210830") == "TIC_307210830"
    
    def test_multiple_spaces(self):
        """Multiple spaces become underscores."""
        assert sanitize_filename("TIC  307210830") == "TIC__307210830"
    
    def test_special_chars_removed(self):
        """Special characters are removed."""
        assert sanitize_filename("TIC@307210830!") == "TIC307210830"
    
    def test_keeps_alphanumeric_underscore_hyphen_dot(self):
        """Allowed characters are preserved."""
        assert sanitize_filename("TIC-307210830_test.v1") == "TIC-307210830_test.v1"
    
    def test_unicode_removed(self):
        """Non-ASCII characters are removed."""
        assert sanitize_filename("TIC_307210830_α") == "TIC_307210830_"


class TestParseCatalog:
    """Tests for parse_catalog function."""

    def test_newline_separated_txt(self):
        """Plain newline-separated .txt file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("307210830\n261136679\n")
            f.flush()
            result = parse_catalog(Path(f.name), "TESS")
        
        assert result == ["TIC 307210830", "TIC 261136679"]
    
    def test_newline_with_comments(self):
        """Lines starting with # are ignored."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Comment line\n307210830\n# Another comment\n261136679\n")
            f.flush()
            result = parse_catalog(Path(f.name), "TESS")
        
        assert result == ["TIC 307210830", "TIC 261136679"]
    
    def test_newline_ignores_empty_lines(self):
        """Empty lines are ignored."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("307210830\n\n261136679\n")
            f.flush()
            result = parse_catalog(Path(f.name), "TESS")
        
        assert result == ["TIC 307210830", "TIC 261136679"]
    
    def test_csv_with_tic_id_header(self):
        """CSV with 'tic_id' header column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("tic_id,ra,dec\n307210830,10.0,20.0\n261136679,11.0,21.0\n")
            f.flush()
            result = parse_catalog(Path(f.name), "TESS")
        
        assert result == ["TIC 307210830", "TIC 261136679"]
    
    def test_csv_with_target_header(self):
        """CSV with 'target' header column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("target,mag\n307210830,12.5\n261136679,13.0\n")
            f.flush()
            result = parse_catalog(Path(f.name), "TESS")
        
        assert result == ["TIC 307210830", "TIC 261136679"]
    
    def test_csv_with_id_header(self):
        """CSV with generic 'id' header column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,name\n307210830,Star1\n261136679,Star2\n")
            f.flush()
            result = parse_catalog(Path(f.name), "TESS")
        
        assert result == ["TIC 307210830", "TIC 261136679"]
    
    def test_csv_kepler_ids(self):
        """CSV with Kepler IDs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("tic_id\n11446443\n11446444\n")
            f.flush()
            result = parse_catalog(Path(f.name), "KEPLER")
        
        assert result == ["KIC 11446443", "KIC 11446444"]
    
    def test_utf8_bom_txt(self):
        """UTF-8 BOM in .txt file is handled."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            f.write(b"\xef\xbb\xbf307210830\n261136679\n")
            f.flush()
            result = parse_catalog(Path(f.name), "TESS")
        
        assert result == ["TIC 307210830", "TIC 261136679"]
    
    def test_utf8_bom_csv(self):
        """UTF-8 BOM in .csv file is handled."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write(b"\xef\xbb\xbftic_id\n307210830\n261136679\n")
            f.flush()
            result = parse_catalog(Path(f.name), "TESS")
        
        assert result == ["TIC 307210830", "TIC 261136679"]
    
    def test_lowercase_tic_in_file(self):
        """Lowercase 'tic' entries in file get normalized."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("tic307210830\ntic 261136679\n")
            f.flush()
            result = parse_catalog(Path(f.name), "TESS")
        
        assert result == ["TIC 307210830", "TIC 261136679"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])