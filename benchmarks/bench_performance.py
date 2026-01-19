"""
Benchmark suite for tldr-code performance.

Run with:
    uv run pytest benchmarks/bench_performance.py -v

For detailed stats:
    uv run pytest benchmarks/bench_performance.py -v --benchmark-only

Compare runs:
    uv run pytest benchmarks/bench_performance.py --benchmark-save=baseline
    uv run pytest benchmarks/bench_performance.py --benchmark-compare=baseline
"""

import pytest
import tempfile
from pathlib import Path

# Import API functions
from tldr.api import (
    get_code_structure,
    search,
    get_cfg_context,
    get_dfg_context,
)
from tldr.analysis import analyze_impact
from tldr.cross_file_calls import build_project_call_graph


# Use the tldr package itself as the benchmark target
TLDR_ROOT = Path(__file__).parent.parent / "tldr"


class TestStructureBenchmarks:
    """Benchmarks for code structure extraction (codemaps)."""

    @pytest.mark.benchmark(group="structure")
    def test_structure_python(self, benchmark):
        """Benchmark: tldr structure . --lang python"""
        result = benchmark(get_code_structure, str(TLDR_ROOT), "python")
        assert result is not None
        assert "files" in result

    @pytest.mark.benchmark(group="structure")
    def test_structure_single_file(self, benchmark):
        """Benchmark: tldr structure api.py --lang python"""
        api_file = TLDR_ROOT / "api.py"
        result = benchmark(get_code_structure, str(api_file.parent), "python", max_results=1)
        assert result is not None


class TestCallGraphBenchmarks:
    """Benchmarks for call graph building."""

    @pytest.mark.benchmark(group="callgraph")
    def test_calls_full_project(self, benchmark):
        """Benchmark: tldr calls . (full call graph)"""
        result = benchmark(build_project_call_graph, str(TLDR_ROOT), "python")
        assert result is not None
        # build_project_call_graph returns a ProjectCallGraph with edges set
        assert len(result.edges) >= 0


class TestSearchBenchmarks:
    """Benchmarks for code search."""

    @pytest.mark.benchmark(group="search")
    def test_search_simple_pattern(self, benchmark):
        """Benchmark: tldr search 'def ' ."""
        result = benchmark(search, r"def \w+", str(TLDR_ROOT))
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.benchmark(group="search")
    def test_search_complex_pattern(self, benchmark):
        """Benchmark: tldr search 'class.*Error' ."""
        result = benchmark(search, r"class \w+Error", str(TLDR_ROOT))
        assert result is not None
        assert isinstance(result, list)

    @pytest.mark.benchmark(group="search")
    def test_search_with_extensions(self, benchmark):
        """Benchmark: tldr search 'import' . --ext .py"""
        result = benchmark(
            search,
            r"^import|^from .* import",
            str(TLDR_ROOT),
            extensions={".py"}
        )
        assert result is not None
        assert isinstance(result, list)


class TestImpactBenchmarks:
    """Benchmarks for impact analysis (reverse call graph)."""

    @pytest.mark.benchmark(group="impact")
    def test_impact_common_function(self, benchmark):
        """Benchmark: tldr impact get_code_structure ."""
        result = benchmark(
            analyze_impact,
            str(TLDR_ROOT),
            "get_code_structure",
            max_depth=2,
            language="python"
        )
        assert result is not None


class TestCFGBenchmarks:
    """Benchmarks for control flow graph generation."""

    @pytest.mark.benchmark(group="cfg")
    def test_cfg_medium_function(self, benchmark):
        """Benchmark: tldr cfg api.py search"""
        api_file = TLDR_ROOT / "api.py"
        result = benchmark(
            get_cfg_context,
            str(api_file),
            "search",
            "python"
        )
        assert result is not None

    @pytest.mark.benchmark(group="cfg")
    def test_cfg_small_function(self, benchmark):
        """Benchmark: tldr cfg api.py extract_file"""
        api_file = TLDR_ROOT / "api.py"
        result = benchmark(
            get_cfg_context,
            str(api_file),
            "extract_file",
            "python"
        )
        assert result is not None


class TestDFGBenchmarks:
    """Benchmarks for data flow graph generation."""

    @pytest.mark.benchmark(group="dfg")
    def test_dfg_medium_function(self, benchmark):
        """Benchmark: tldr dfg api.py search"""
        api_file = TLDR_ROOT / "api.py"
        result = benchmark(
            get_dfg_context,
            str(api_file),
            "search",
            "python"
        )
        assert result is not None

    @pytest.mark.benchmark(group="dfg")
    def test_dfg_small_function(self, benchmark):
        """Benchmark: tldr dfg api.py extract_file"""
        api_file = TLDR_ROOT / "api.py"
        result = benchmark(
            get_dfg_context,
            str(api_file),
            "extract_file",
            "python"
        )
        assert result is not None


class TestSyntheticBenchmarks:
    """Benchmarks on synthetic codebases of controlled sizes."""

    @staticmethod
    def _create_synthetic_project(tmpdir: Path, num_files: int, funcs_per_file: int) -> Path:
        """Create a synthetic Python project for benchmarking."""
        for i in range(num_files):
            file_path = tmpdir / f"module_{i}.py"
            lines = [f'"""Module {i}."""\n']
            for j in range(funcs_per_file):
                lines.append(f"""
def func_{i}_{j}(x, y):
    '''Function {j} in module {i}.'''
    result = x + y
    if result > 10:
        return result * 2
    else:
        return result
""")
            file_path.write_text("".join(lines))
        return tmpdir

    @pytest.mark.benchmark(group="synthetic")
    def test_structure_10_files(self, benchmark, tmp_path):
        """Benchmark: structure on 10 files, 10 funcs each"""
        project = self._create_synthetic_project(tmp_path, 10, 10)
        result = benchmark(get_code_structure, str(project), "python")
        assert len(result["files"]) == 10

    @pytest.mark.benchmark(group="synthetic")
    def test_structure_50_files(self, benchmark, tmp_path):
        """Benchmark: structure on 50 files, 5 funcs each"""
        project = self._create_synthetic_project(tmp_path, 50, 5)
        result = benchmark(get_code_structure, str(project), "python")
        assert len(result["files"]) == 50

    @pytest.mark.benchmark(group="synthetic")
    def test_calls_10_files(self, benchmark, tmp_path):
        """Benchmark: call graph on 10 files"""
        project = self._create_synthetic_project(tmp_path, 10, 10)
        result = benchmark(build_project_call_graph, str(project), "python")
        assert result is not None

    @pytest.mark.benchmark(group="synthetic")
    def test_search_50_files(self, benchmark, tmp_path):
        """Benchmark: search across 50 files"""
        project = self._create_synthetic_project(tmp_path, 50, 5)
        result = benchmark(search, r"def func_", str(project), max_results=300)
        assert result is not None
        assert len(result) >= 250  # 50 files * 5 funcs


# Parametrized benchmarks for different scales
@pytest.mark.parametrize("num_files,funcs_per_file", [
    (5, 5),
    (10, 10),
    (25, 5),
])
class TestScaleBenchmarks:
    """Benchmarks at different scales to measure complexity."""

    @staticmethod
    def _create_project(tmpdir: Path, num_files: int, funcs_per_file: int) -> Path:
        """Create synthetic project."""
        for i in range(num_files):
            file_path = tmpdir / f"mod_{i}.py"
            content = f'"""Module {i}."""\n'
            for j in range(funcs_per_file):
                content += f"""
def f_{i}_{j}(a):
    return a + {j}
"""
            file_path.write_text(content)
        return tmpdir

    @pytest.mark.benchmark(group="scale-structure")
    def test_scale_structure(self, benchmark, tmp_path, num_files, funcs_per_file):
        """Benchmark structure at different scales."""
        project = self._create_project(tmp_path, num_files, funcs_per_file)
        result = benchmark(get_code_structure, str(project), "python")
        assert len(result["files"]) == num_files

    @pytest.mark.benchmark(group="scale-calls")
    def test_scale_calls(self, benchmark, tmp_path, num_files, funcs_per_file):
        """Benchmark call graph at different scales."""
        project = self._create_project(tmp_path, num_files, funcs_per_file)
        result = benchmark(build_project_call_graph, str(project), "python")
        assert result is not None
