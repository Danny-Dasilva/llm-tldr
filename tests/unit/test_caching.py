"""Tests for LRU caching on hot functions."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestScanProjectCaching:
    """Test that scan_project uses caching properly."""

    def test_scan_project_returns_same_result_on_repeated_calls(self, tmp_path: Path):
        """Repeated calls with same args should return identical results."""
        from tldr.cross_file_calls import scan_project

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("# test")

        result1 = scan_project(str(tmp_path), language="python")
        result2 = scan_project(str(tmp_path), language="python")

        assert result1 == result2
        assert len(result1) == 1

    def test_scan_project_cache_info_exists(self):
        """_scan_project_cached should have cache_info method from lru_cache."""
        from tldr.cross_file_calls import _scan_project_cached

        # The cached function should have cache_info
        assert hasattr(_scan_project_cached, "cache_info")
        assert hasattr(_scan_project_cached, "cache_clear")

    def test_scan_project_cache_hit_on_same_args(self, tmp_path: Path):
        """Cache should hit on identical arguments."""
        from tldr.cross_file_calls import _scan_project_cached

        # Clear any existing cache
        _scan_project_cached.cache_clear()

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("# test")

        # First call - cache miss
        _scan_project_cached(str(tmp_path), "python", None, True)
        info1 = _scan_project_cached.cache_info()

        # Second call - cache hit
        _scan_project_cached(str(tmp_path), "python", None, True)
        info2 = _scan_project_cached.cache_info()

        assert info2.hits > info1.hits


class TestParseImportsCaching:
    """Test that parse_imports uses caching properly."""

    def test_parse_imports_returns_same_result_on_repeated_calls(self, tmp_path: Path):
        """Repeated calls with same args should return identical results."""
        from tldr.cross_file_calls import parse_imports

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("import os\nfrom pathlib import Path")

        result1 = parse_imports(test_file)
        result2 = parse_imports(test_file)

        assert result1 == result2
        assert len(result1) == 2

    def test_parse_imports_cache_info_exists(self):
        """_parse_imports_cached should have cache_info method from lru_cache."""
        from tldr.cross_file_calls import _parse_imports_cached

        # The cached function should have cache_info from lru_cache
        assert hasattr(_parse_imports_cached, "cache_info")
        assert hasattr(_parse_imports_cached, "cache_clear")

    def test_parse_imports_cache_hit_on_same_file(self, tmp_path: Path):
        """Cache should hit when parsing same file."""
        from tldr.cross_file_calls import _parse_imports_cached

        # Clear any existing cache
        _parse_imports_cached.cache_clear()

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("import os")

        # First call - cache miss
        _parse_imports_cached(str(test_file))
        info1 = _parse_imports_cached.cache_info()

        # Second call - cache hit
        _parse_imports_cached(str(test_file))
        info2 = _parse_imports_cached.cache_info()

        assert info2.hits > info1.hits


class TestLanguageCaching:
    """Test that Language objects are cached."""

    def test_parser_cache_exists(self):
        """Module-level parser cache should exist."""
        from tldr import cross_file_calls

        assert hasattr(cross_file_calls, "_parser_cache")
        assert isinstance(cross_file_calls._parser_cache, dict)

    def test_get_ts_parser_caches_parser(self):
        """_get_ts_parser should return cached parser on second call."""
        pytest.importorskip("tree_sitter_typescript")

        from tldr.cross_file_calls import _get_ts_parser, _parser_cache

        # Clear cache for test isolation
        _parser_cache.clear()

        parser1 = _get_ts_parser()
        parser2 = _get_ts_parser()

        # Same object (cached)
        assert parser1 is parser2
        assert "typescript" in _parser_cache

    def test_get_go_parser_caches_parser(self):
        """_get_go_parser should return cached parser on second call."""
        pytest.importorskip("tree_sitter_go")

        from tldr.cross_file_calls import _get_go_parser, _parser_cache

        # Clear cache for test isolation
        _parser_cache.clear()

        parser1 = _get_go_parser()
        parser2 = _get_go_parser()

        # Same object (cached)
        assert parser1 is parser2
        assert "go" in _parser_cache

    def test_get_rust_parser_caches_parser(self):
        """_get_rust_parser should return cached parser on second call."""
        pytest.importorskip("tree_sitter_rust")

        from tldr.cross_file_calls import _get_rust_parser, _parser_cache

        # Clear cache for test isolation
        _parser_cache.clear()

        parser1 = _get_rust_parser()
        parser2 = _get_rust_parser()

        # Same object (cached)
        assert parser1 is parser2
        assert "rust" in _parser_cache
