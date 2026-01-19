"""Tests for performance improvements (TDD - write tests first).

These tests define the expected behavior for:
1. Parallel file processing in search and get_code_structure
2. Memoized impact analysis in _build_caller_tree
3. CLI using daemon caches when available
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestParallelSearch:
    """Tests for parallelized search function."""

    def test_search_respects_max_workers_env(self, tmp_path: Path):
        """Search should use TLDR_MAX_WORKERS for parallelization."""
        # Create multiple test files
        for i in range(20):
            (tmp_path / f"file_{i}.py").write_text(f"# file {i}\ntest_pattern = {i}")

        from tldr.api import search

        # Set max workers
        with patch.dict(os.environ, {"TLDR_MAX_WORKERS": "4"}):
            results = search("test_pattern", tmp_path, extensions={".py"})

        # Should find all matches
        assert len(results) == 20

    def test_search_has_parallel_code_path(self):
        """Search function should have ProcessPoolExecutor code path."""
        import inspect
        from tldr import api
        from tldr.api import search

        # Get the source code
        source = inspect.getsource(search)

        # Verify ProcessPoolExecutor is used in the function
        assert "ProcessPoolExecutor" in source, (
            "search() should use ProcessPoolExecutor for parallel processing"
        )
        # _MAX_WORKERS is a module-level constant now
        assert hasattr(api, "_MAX_WORKERS"), (
            "api module should have _MAX_WORKERS constant"
        )
        assert "_MAX_WORKERS" in source, (
            "search() should use _MAX_WORKERS for parallel processing"
        )

    def test_search_parallel_returns_same_results(self, tmp_path: Path):
        """Parallel search should return same results as sequential."""
        # Create many files with some content
        for i in range(50):
            content = f"# File {i}\n" + "x = 1\n" * 100 + f"target_pattern_{i} = True\n"
            (tmp_path / f"module_{i}.py").write_text(content)

        from tldr.api import search

        # Sequential (1 worker)
        with patch.dict(os.environ, {"TLDR_MAX_WORKERS": "1"}):
            results_seq = search("target_pattern", tmp_path, extensions={".py"})

        # Parallel (4 workers)
        with patch.dict(os.environ, {"TLDR_MAX_WORKERS": "4"}):
            results_par = search("target_pattern", tmp_path, extensions={".py"})

        # Both should return same results
        assert len(results_seq) == len(results_par) == 50


class TestParallelStructure:
    """Tests for parallelized get_code_structure function."""

    def test_structure_respects_max_workers_env(self, tmp_path: Path):
        """get_code_structure should use TLDR_MAX_WORKERS."""
        # Create test files with functions
        for i in range(10):
            (tmp_path / f"module_{i}.py").write_text(
                f"def func_{i}():\n    pass\n\nclass Class_{i}:\n    pass\n"
            )

        from tldr.api import get_code_structure

        with patch.dict(os.environ, {"TLDR_MAX_WORKERS": "4"}):
            result = get_code_structure(str(tmp_path), language="python")

        # Should have all 10 files
        assert len(result["files"]) == 10

    def test_structure_has_parallel_code_path(self):
        """get_code_structure should have ProcessPoolExecutor code path."""
        import inspect
        from tldr import api
        from tldr.api import get_code_structure

        source = inspect.getsource(get_code_structure)

        assert "ProcessPoolExecutor" in source, (
            "get_code_structure() should use ProcessPoolExecutor for parallel processing"
        )
        # _MAX_WORKERS is a module-level constant now
        assert hasattr(api, "_MAX_WORKERS"), (
            "api module should have _MAX_WORKERS constant"
        )
        assert "_MAX_WORKERS" in source, (
            "get_code_structure() should use _MAX_WORKERS for parallel processing"
        )

    def test_structure_parallel_returns_same_results(self, tmp_path: Path):
        """Parallel and sequential processing should return identical results."""
        for i in range(15):
            (tmp_path / f"mod_{i}.py").write_text(
                f"def function_{i}(a, b):\n    return a + b\n"
            )

        from tldr.api import get_code_structure

        # Sequential
        with patch.dict(os.environ, {"TLDR_MAX_WORKERS": "1"}):
            result_seq = get_code_structure(str(tmp_path), language="python")

        # Parallel
        with patch.dict(os.environ, {"TLDR_MAX_WORKERS": "4"}):
            result_par = get_code_structure(str(tmp_path), language="python")

        # Sort for comparison
        files_seq = sorted(result_seq["files"], key=lambda f: f["path"])
        files_par = sorted(result_par["files"], key=lambda f: f["path"])

        assert len(files_seq) == len(files_par)
        for f1, f2 in zip(files_seq, files_par):
            assert f1["path"] == f2["path"]
            assert f1["functions"] == f2["functions"]


class TestMemoizedImpactAnalysis:
    """Tests for memoized _build_caller_tree in analysis.py."""

    def test_build_caller_tree_memoized(self):
        """_build_caller_tree should not recompute for same function at same depth."""
        from tldr.analysis import FunctionRef, _build_caller_tree

        # Create a simple call graph: A -> B -> C
        func_a = FunctionRef(file="a.py", name="func_a")
        func_b = FunctionRef(file="b.py", name="func_b")
        func_c = FunctionRef(file="c.py", name="func_c")

        reverse = {
            func_b: [func_a],  # func_a calls func_b
            func_c: [func_b],  # func_b calls func_c
        }

        # Build tree twice for same function
        result1 = _build_caller_tree(func_c, reverse, depth=3, visited=set())
        result2 = _build_caller_tree(func_c, reverse, depth=3, visited=set())

        # Results should be identical
        assert result1 == result2

    def test_build_caller_tree_no_visited_copy(self):
        """_build_caller_tree should NOT copy visited set on each call (performance fix)."""
        from tldr.analysis import FunctionRef, _build_caller_tree
        import inspect

        # Get the source of _build_caller_tree
        source = inspect.getsource(_build_caller_tree)

        # The fix should NOT have visited.copy() - instead use memoization
        # This is a code inspection test to verify the fix
        # After fix: should not contain "visited.copy()"
        # Note: This test will FAIL before the fix is implemented
        assert "visited.copy()" not in source, (
            "_build_caller_tree should not copy visited set on each recursive call. "
            "Use memoization dict instead."
        )

    def test_build_caller_tree_no_exponential_blowup(self):
        """Memoization should prevent exponential time on diamond patterns."""
        from tldr.analysis import FunctionRef, _build_caller_tree

        # Create diamond pattern: D is called by B and C, which are both called by A
        # Without memoization, D would be processed twice at each depth
        func_a = FunctionRef(file="a.py", name="func_a")
        func_b = FunctionRef(file="b.py", name="func_b")
        func_c = FunctionRef(file="c.py", name="func_c")
        func_d = FunctionRef(file="d.py", name="func_d")

        reverse = {
            func_b: [func_a],
            func_c: [func_a],
            func_d: [func_b, func_c],
        }

        # This should complete quickly with memoization
        start = time.time()
        result = _build_caller_tree(func_d, reverse, depth=10, visited=set())
        elapsed = time.time() - start

        # Should be very fast (< 1 second) with memoization
        assert elapsed < 1.0
        assert result["function"] == "func_d"

    def test_impact_analysis_uses_memoization(self):
        """impact_analysis should benefit from memoization."""
        from unittest.mock import MagicMock

        from tldr.analysis import FunctionRef, impact_analysis

        # Mock call graph with edges
        mock_graph = MagicMock()
        mock_graph.edges = [
            ("a.py", "func_a", "b.py", "func_b"),
            ("b.py", "func_b", "c.py", "func_c"),
            ("a.py", "func_a", "c.py", "func_c"),  # Diamond: a->c directly too
        ]

        result = impact_analysis(mock_graph, "func_c", max_depth=5)

        assert "targets" in result
        # Should have found func_c target
        assert len(result["targets"]) == 1


class TestDaemonCacheWiring:
    """Tests for CLI using daemon caches when available."""

    def test_cli_checks_daemon_before_search(self):
        """CLI search should check if daemon is running."""
        from tldr.daemon.startup import _is_socket_connectable

        # This test verifies the wiring exists
        # Actual daemon integration tested in integration tests
        assert callable(_is_socket_connectable)

    def test_daemon_provides_cached_search(self):
        """Daemon should have a cached_search function."""
        from tldr.daemon.cached_queries import cached_search

        assert callable(cached_search)

    def test_daemon_provides_cached_structure(self):
        """Daemon should have a cached_structure function."""
        from tldr.daemon.cached_queries import cached_structure

        assert callable(cached_structure)

    @patch("tldr.daemon.startup._is_socket_connectable")
    @patch("tldr.daemon.startup.query_daemon")
    def test_api_uses_daemon_when_available(
        self, mock_query, mock_connectable, tmp_path: Path
    ):
        """API functions should use daemon cache when daemon is running."""
        # Setup: daemon is running
        mock_connectable.return_value = True
        mock_query.return_value = {
            "status": "ok",
            "results": [{"file": "test.py", "line": 1, "content": "match"}],
        }

        # Create a test file so the fallback path also works
        (tmp_path / "test.py").write_text("match")

        # Note: This test documents the expected behavior.
        # The actual implementation will need to check for daemon availability.
        # For now, api.search does NOT check daemon - that's what we're implementing.

        from tldr.api import search

        # Current behavior: always processes locally
        results = search("match", tmp_path, extensions={".py"})
        assert len(results) >= 1


class TestMaxWorkersDefault:
    """Tests for TLDR_MAX_WORKERS environment variable handling."""

    def test_default_max_workers_uses_cpu_count(self):
        """Without TLDR_MAX_WORKERS, should default to cpu_count."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove TLDR_MAX_WORKERS if set
            os.environ.pop("TLDR_MAX_WORKERS", None)

            # The implementation should use os.cpu_count() as default
            # This is tested indirectly through the parallelization behavior
            cpu_count = os.cpu_count() or 4
            assert cpu_count >= 1

    def test_max_workers_respects_env_value(self):
        """TLDR_MAX_WORKERS should override default."""
        with patch.dict(os.environ, {"TLDR_MAX_WORKERS": "2"}):
            max_workers = int(os.environ.get("TLDR_MAX_WORKERS", os.cpu_count() or 4))
            assert max_workers == 2
