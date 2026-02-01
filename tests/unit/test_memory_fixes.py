"""Tests for memory leak fixes.

Tests for:
1. Mutex/lock for background reindex (Fix 1)
2. .tldrignore check in _handle_notify (Fix 2)
3. Stream-based file hashing (Fix 3)
4. Cache size limits in SalsaDB (Fix 4)
"""

import hashlib
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestStreamBasedFileHashing:
    """Fix 3: Stream-based file hashing to avoid loading large files into memory."""

    def test_compute_file_hash_small_file(self):
        """Small files should produce the same hash as before."""
        from tldr.patch import compute_file_hash

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            content = b"Hello, World!"
            f.write(content)
            f.flush()
            path = f.name

        try:
            result = compute_file_hash(path)
            # SHA-1 of "Hello, World!" (without newline)
            expected = hashlib.sha1(content).hexdigest()
            assert result == expected, f"Expected {expected}, got {result}"
            assert len(result) == 40, "SHA-1 should be 40 hex chars"
        finally:
            os.unlink(path)

    def test_compute_file_hash_large_file(self):
        """Large files should be hashed correctly via streaming."""
        from tldr.patch import compute_file_hash

        # Create a 5MB file (larger than typical chunk size)
        chunk_size = 65536
        file_size = 5 * 1024 * 1024  # 5MB

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            # Write deterministic content
            content = b"x" * file_size
            f.write(content)
            f.flush()
            path = f.name

        try:
            result = compute_file_hash(path)
            expected = hashlib.sha1(content).hexdigest()
            assert result == expected, f"Expected {expected}, got {result}"
        finally:
            os.unlink(path)

    def test_compute_file_hash_does_not_load_full_file(self):
        """Verify that large files are not fully loaded into memory."""
        # This is a behavioral test - the new implementation should use
        # chunked reading instead of path.read_bytes()
        from tldr import patch
        import inspect

        source = inspect.getsource(patch.compute_file_hash)

        # Old implementation uses: path.read_bytes()
        # New implementation should NOT have this for large files
        # Instead, it should have a while loop reading chunks
        assert "while" in source or "chunk" in source.lower(), (
            "compute_file_hash should use chunked reading for memory efficiency"
        )


class TestSalsaDBCacheLimits:
    """Fix 4: SalsaDB cache should have size limits with LRU eviction."""

    def test_salsa_db_has_max_size(self):
        """SalsaDB should have a configurable max cache size."""
        from tldr.salsa import SalsaDB

        db = SalsaDB()
        # Should have a max_size attribute or similar
        assert hasattr(db, "_max_cache_size") or hasattr(db, "max_cache_size"), (
            "SalsaDB should have a max cache size attribute"
        )

    def test_salsa_db_evicts_old_entries(self):
        """Cache should evict entries when exceeding max size."""
        from tldr.salsa import SalsaDB

        # Create DB with small max size for testing
        db = SalsaDB(max_cache_size=5)

        # Add 10 entries (more than max)
        def dummy_query(x):
            return x * 2

        for i in range(10):
            db.query(dummy_query, i)

        # Should have evicted older entries
        cache_size = len(db._query_cache)
        assert cache_size <= 5, f"Cache size {cache_size} exceeds max 5"

    def test_salsa_db_default_max_size(self):
        """Default max size should be reasonable (e.g., 10000)."""
        from tldr.salsa import SalsaDB

        db = SalsaDB()
        max_size = getattr(db, "_max_cache_size", getattr(db, "max_cache_size", 0))
        assert max_size >= 1000, f"Default max size {max_size} seems too small"
        assert max_size <= 100000, f"Default max size {max_size} seems too large"


class TestBackgroundReindexLock:
    """Fix 1: Only one background reindex should run at a time."""

    def test_reindex_lock_prevents_concurrent_runs(self):
        """Multiple calls to _trigger_background_reindex should not spawn multiple processes."""
        from tldr.daemon.core import TLDRDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon = TLDRDaemon(Path(tmpdir))
            daemon._dirty_files = {"file1.py", "file2.py"}
            daemon._dirty_count = 2

            # Track how many times the subprocess would be spawned
            spawn_count = 0
            original_run = None

            def mock_subprocess_run(*args, **kwargs):
                nonlocal spawn_count
                spawn_count += 1
                time.sleep(0.1)  # Simulate some work
                return MagicMock(returncode=0, stderr="")

            with patch("subprocess.run", side_effect=mock_subprocess_run):
                # Call trigger multiple times rapidly
                threads = []
                for _ in range(5):
                    t = threading.Thread(target=daemon._trigger_background_reindex)
                    threads.append(t)
                    t.start()

                # Wait for all threads to complete
                for t in threads:
                    t.join(timeout=2)

            # Should only have spawned once due to lock
            assert spawn_count <= 1, f"Expected 1 subprocess, but {spawn_count} were spawned"

    def test_reindex_in_progress_flag_set_atomically(self):
        """The _reindex_in_progress flag should be thread-safe."""
        from tldr.daemon.core import TLDRDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon = TLDRDaemon(Path(tmpdir))

            # Should have a lock for reindex operations
            assert hasattr(daemon, "_reindex_lock"), (
                "TLDRDaemon should have a _reindex_lock for thread safety"
            )


class TestHandleNotifyIgnoreCheck:
    """Fix 2: _handle_notify should check .tldrignore EARLY."""

    def test_handle_notify_ignores_binary_files(self):
        """Binary files (.whl, .so) should be ignored in _handle_notify."""
        from tldr.daemon.core import TLDRDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .tldrignore that ignores .whl and .so files
            tldrignore = Path(tmpdir) / ".tldrignore"
            tldrignore.write_text("*.whl\n*.so\n")

            daemon = TLDRDaemon(Path(tmpdir))

            # Notify about a .whl file
            result = daemon._handle_notify({"file": str(Path(tmpdir) / "package.whl")})

            # Should NOT add to dirty files
            assert daemon._dirty_count == 0, (
                f"Binary file should not increment dirty count (got {daemon._dirty_count})"
            )
            assert result.get("ignored", False) or result.get("status") == "ignored", (
                "Response should indicate file was ignored"
            )

    def test_handle_notify_allows_source_files(self):
        """Source files that pass .tldrignore should be processed."""
        from tldr.daemon.core import TLDRDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .tldrignore that only ignores .whl files
            tldrignore = Path(tmpdir) / ".tldrignore"
            tldrignore.write_text("*.whl\n")

            # Create a Python file
            py_file = Path(tmpdir) / "module.py"
            py_file.write_text("print('hello')")

            daemon = TLDRDaemon(Path(tmpdir))

            # Notify about a .py file
            result = daemon._handle_notify({"file": str(py_file)})

            # Should add to dirty files
            assert daemon._dirty_count == 1, (
                f"Source file should increment dirty count (got {daemon._dirty_count})"
            )

    def test_handle_notify_checks_ignore_before_processing(self):
        """The ignore check should happen EARLY, before any expensive operations."""
        from tldr.daemon.core import TLDRDaemon
        import inspect

        source = inspect.getsource(TLDRDaemon._handle_notify)

        # The ignore check should happen near the start of the function
        # Look for should_ignore call before dirty_files tracking
        should_ignore_pos = source.find("should_ignore")
        dirty_files_pos = source.find("_dirty_files")

        assert should_ignore_pos != -1, (
            "_handle_notify should call should_ignore to filter files"
        )
        assert should_ignore_pos < dirty_files_pos, (
            "should_ignore check should come BEFORE dirty file tracking"
        )


class TestMemoryPressureHandler:
    """Test the memory pressure handler in TLDRDaemon."""

    def test_constant_exists(self):
        """MEMORY_CLEANUP_THRESHOLD_GB constant should exist."""
        from tldr.daemon.core import MEMORY_CLEANUP_THRESHOLD_GB

        assert MEMORY_CLEANUP_THRESHOLD_GB == 3.0

    def test_check_memory_pressure_method_exists(self):
        """TLDRDaemon should have a _check_memory_pressure method."""
        from tldr.daemon.core import TLDRDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon = TLDRDaemon(Path(tmpdir))
            assert hasattr(daemon, "_check_memory_pressure")
            assert callable(daemon._check_memory_pressure)

    def test_no_cleanup_below_threshold(self):
        """When RSS is below threshold, no cleanup should happen."""
        from tldr.daemon.core import TLDRDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon = TLDRDaemon(Path(tmpdir))

            with patch("tldr.daemon.core._get_rss_gb", return_value=1.0):
                with patch("tldr.daemon.core.gc.collect") as mock_gc:
                    daemon._check_memory_pressure()
                    mock_gc.assert_not_called()

    def test_cleanup_above_threshold(self):
        """When RSS exceeds threshold, caches should be cleared and gc.collect called."""
        from tldr.daemon.core import TLDRDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon = TLDRDaemon(Path(tmpdir))

            with patch("tldr.daemon.core._get_rss_gb", return_value=4.0):
                with patch("tldr.daemon.core.gc.collect") as mock_gc:
                    with patch(
                        "tldr.daemon.core._scan_project_cached"
                    ) as mock_scan:
                        mock_scan.cache_clear = MagicMock()
                        with patch(
                            "tldr.daemon.core._parse_imports_cached"
                        ) as mock_parse:
                            mock_parse.cache_clear = MagicMock()
                            daemon._check_memory_pressure()
                            mock_scan.cache_clear.assert_called_once()
                            mock_parse.cache_clear.assert_called_once()
                            mock_gc.assert_called_once()

    def test_check_memory_pressure_called_in_handle_notify(self):
        """_check_memory_pressure should be called at the end of _handle_notify."""
        from tldr.daemon.core import TLDRDaemon
        import inspect

        source = inspect.getsource(TLDRDaemon._handle_notify)
        assert "_check_memory_pressure" in source, (
            "_handle_notify should call _check_memory_pressure"
        )


class TestHashingMemoryEfficiency:
    """Verify the hashing implementation is memory-efficient."""

    def test_hash_function_signature(self):
        """compute_file_hash should accept optional chunk_size parameter."""
        from tldr.patch import compute_file_hash
        import inspect

        sig = inspect.signature(compute_file_hash)
        params = list(sig.parameters.keys())

        # Should have chunk_size parameter (optional)
        assert "chunk_size" in params or len(params) == 1, (
            "compute_file_hash should either have chunk_size param or use internal default"
        )
