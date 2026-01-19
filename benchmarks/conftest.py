"""Pytest configuration for benchmarks."""

import pytest


def pytest_configure(config):
    """Configure pytest for benchmark runs."""
    # Add benchmark markers
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark"
    )


@pytest.fixture(scope="session")
def tldr_root():
    """Return the path to the tldr package for benchmarking."""
    from pathlib import Path
    return Path(__file__).parent.parent / "tldr"
