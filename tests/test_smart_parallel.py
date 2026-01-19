"""Tests for smart parallelization threshold in api.py.

Validates that ProcessPoolExecutor is only used above the threshold.
"""

import os
from pathlib import Path
from unittest import mock

import pytest


class TestSmartParallelThreshold:
    """Tests for threshold-based parallelization in search() and get_code_structure()."""

    @pytest.fixture
    def small_project(self, tmp_path: Path):
        """Create a small project with <100 files (below threshold)."""
        src = tmp_path / "src"
        src.mkdir()
        for i in range(10):  # 10 files - below threshold
            (src / f"file_{i}.py").write_text(f"def func_{i}(): pass\n")
        return tmp_path

    # =========================================================================
    # Threshold Configuration Tests
    # =========================================================================

    def test_parallel_threshold_default(self):
        """Threshold defaults to 100 if env var not set."""
        from tldr import api

        # Clear env var if set
        with mock.patch.dict(os.environ, {}, clear=True):
            # Re-import to pick up default
            import importlib
            importlib.reload(api)
            assert api._PARALLEL_THRESHOLD == 100

    def test_parallel_threshold_env_override(self):
        """Threshold can be overridden via TLDR_PARALLEL_THRESHOLD."""
        from tldr import api

        with mock.patch.dict(os.environ, {"TLDR_PARALLEL_THRESHOLD": "50"}):
            import importlib
            importlib.reload(api)
            assert api._PARALLEL_THRESHOLD == 50

    def test_max_workers_default(self):
        """Max workers defaults to 4 if env var not set."""
        from tldr import api

        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            importlib.reload(api)
            assert api._MAX_WORKERS == 4

    def test_max_workers_env_override(self):
        """Max workers can be overridden via TLDR_MAX_WORKERS."""
        from tldr import api

        with mock.patch.dict(os.environ, {"TLDR_MAX_WORKERS": "8"}):
            import importlib
            importlib.reload(api)
            assert api._MAX_WORKERS == 8

    # =========================================================================
    # search() Parallelization Tests
    # =========================================================================

    def test_search_small_project_no_parallelization(self, small_project: Path):
        """search() uses sequential processing for small projects."""
        from tldr import api

        with mock.patch("tldr.api.ProcessPoolExecutor") as mock_executor:
            results = api.search("func_", small_project)
            # Should NOT use ProcessPoolExecutor for small projects
            mock_executor.assert_not_called()
            # Should still return results
            assert len(results) == 10

    def test_search_respects_threshold_boundary(self, tmp_path: Path):
        """search() respects exact threshold boundary - below threshold is sequential."""
        from tldr import api
        import importlib

        # Use a lower threshold for testing
        with mock.patch.dict(os.environ, {"TLDR_PARALLEL_THRESHOLD": "20"}):
            importlib.reload(api)

            # Create 19 files (below threshold of 20)
            src = tmp_path / "src"
            src.mkdir()
            for i in range(19):
                (src / f"file_{i}.py").write_text(f"# test_{i}\n")

            with mock.patch("tldr.api.ProcessPoolExecutor") as mock_executor:
                api.search("test_", tmp_path)
                # Below threshold, should NOT parallelize
                mock_executor.assert_not_called()

    def test_search_above_threshold_calls_executor(self, tmp_path: Path):
        """search() creates ProcessPoolExecutor above threshold."""
        from tldr import api
        import importlib

        # Use a lower threshold for testing
        with mock.patch.dict(os.environ, {"TLDR_PARALLEL_THRESHOLD": "20"}):
            importlib.reload(api)

            # Create 21 files (above threshold of 20)
            src = tmp_path / "src"
            src.mkdir()
            for i in range(21):
                (src / f"file_{i}.py").write_text(f"# test_{i}\n")

            # Mock the executor to verify it's called
            mock_executor_instance = mock.MagicMock()
            mock_executor_instance.__enter__ = mock.MagicMock(return_value=mock_executor_instance)
            mock_executor_instance.__exit__ = mock.MagicMock(return_value=False)
            mock_executor_instance.submit = mock.MagicMock(return_value=mock.MagicMock())

            with mock.patch("tldr.api.ProcessPoolExecutor", return_value=mock_executor_instance) as mock_executor:
                with mock.patch("tldr.api.as_completed", return_value=[]):
                    api.search("test_", tmp_path)
                    # Above threshold, SHOULD call ProcessPoolExecutor
                    mock_executor.assert_called_once()

    # =========================================================================
    # get_code_structure() Parallelization Tests
    # =========================================================================

    def test_structure_small_project_no_parallelization(self, small_project: Path):
        """get_code_structure() uses sequential processing for small projects."""
        from tldr import api

        with mock.patch("tldr.api.ProcessPoolExecutor") as mock_executor:
            result = api.get_code_structure(small_project, language="python")
            # Should NOT use ProcessPoolExecutor for small projects
            mock_executor.assert_not_called()
            # Should still return results
            assert len(result["files"]) == 10

    def test_structure_above_threshold_calls_executor(self, tmp_path: Path):
        """get_code_structure() creates ProcessPoolExecutor above threshold."""
        from tldr import api
        import importlib

        # Use a lower threshold for testing
        with mock.patch.dict(os.environ, {"TLDR_PARALLEL_THRESHOLD": "20"}):
            importlib.reload(api)

            # Create 21 files (above threshold of 20)
            src = tmp_path / "src"
            src.mkdir()
            for i in range(21):
                (src / f"file_{i}.py").write_text(f"def func_{i}(): pass\n")

            # Mock the executor to verify it's called
            mock_executor_instance = mock.MagicMock()
            mock_executor_instance.__enter__ = mock.MagicMock(return_value=mock_executor_instance)
            mock_executor_instance.__exit__ = mock.MagicMock(return_value=False)
            mock_executor_instance.submit = mock.MagicMock(return_value=mock.MagicMock())

            with mock.patch("tldr.api.ProcessPoolExecutor", return_value=mock_executor_instance) as mock_executor:
                with mock.patch("tldr.api.as_completed", return_value=[]):
                    api.get_code_structure(tmp_path, language="python")
                    # Above threshold, SHOULD call ProcessPoolExecutor
                    mock_executor.assert_called_once()

    # =========================================================================
    # Helper Function Tests
    # =========================================================================

    def test_search_file_helper_returns_matches(self, tmp_path: Path):
        """_search_file helper returns correct matches."""
        from tldr.api import _search_file

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\ndef world():\n    pass\n")

        results = _search_file(test_file, "def", tmp_path)
        assert len(results) == 2
        assert results[0]["line"] == 1
        assert results[1]["line"] == 3

    def test_search_file_helper_handles_bad_encoding(self, tmp_path: Path):
        """_search_file helper handles files with bad encoding."""
        from tldr.api import _search_file

        test_file = tmp_path / "test.py"
        test_file.write_bytes(b"\xff\xfe invalid utf-8")

        results = _search_file(test_file, "def", tmp_path)
        # Should return empty list, not crash
        assert results == []

    def test_extract_file_structure_helper(self, tmp_path: Path):
        """_extract_file_structure helper extracts structure correctly."""
        from tldr.api import _extract_file_structure

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n\nclass MyClass:\n    def method(self):\n        pass\n")

        result = _extract_file_structure(test_file, tmp_path)
        assert result is not None
        assert result["path"] == "test.py"
        assert "hello" in result["functions"]
        assert "MyClass" in result["classes"]

    def test_extract_file_structure_handles_errors(self, tmp_path: Path):
        """_extract_file_structure helper handles parse errors gracefully."""
        from tldr.api import _extract_file_structure

        test_file = tmp_path / "test.py"
        test_file.write_bytes(b"\xff\xfe invalid")

        result = _extract_file_structure(test_file, tmp_path)
        # Should not crash - may return empty structure or None
        # The extractor handles bad encoding gracefully
        assert result is None or isinstance(result, dict)

    # =========================================================================
    # Sequential Path Tests
    # =========================================================================

    def test_sequential_search_produces_correct_results(self, small_project: Path):
        """Sequential search path produces correct results."""
        from tldr import api

        results = api.search("func_", small_project)
        assert len(results) == 10
        # Verify all results are valid
        for r in results:
            assert "file" in r
            assert "line" in r
            assert "content" in r

    def test_sequential_structure_produces_correct_results(self, small_project: Path):
        """Sequential structure path produces correct results."""
        from tldr import api

        result = api.get_code_structure(small_project, language="python")
        assert len(result["files"]) == 10
        # Verify all files have expected structure
        for f in result["files"]:
            assert "path" in f
            assert "functions" in f
            assert "classes" in f

    def test_search_respects_max_results_sequential(self, tmp_path: Path):
        """Sequential search respects max_results limit."""
        from tldr import api

        # Create 50 files
        src = tmp_path / "src"
        src.mkdir()
        for i in range(50):
            (src / f"file_{i}.py").write_text(f"# match_{i}\n")

        results = api.search("match_", tmp_path, max_results=10)
        assert len(results) == 10

    def test_structure_respects_max_results_sequential(self, tmp_path: Path):
        """Sequential structure respects max_results limit."""
        from tldr import api

        # Create 50 files
        src = tmp_path / "src"
        src.mkdir()
        for i in range(50):
            (src / f"file_{i}.py").write_text(f"def func_{i}(): pass\n")

        result = api.get_code_structure(tmp_path, language="python", max_results=10)
        assert len(result["files"]) == 10
