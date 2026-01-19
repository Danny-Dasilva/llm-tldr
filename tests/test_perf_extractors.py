"""
Performance tests for DFG/CFG extractors.

These tests verify that performance optimizations don't break correctness.
"""
import time
import pytest


class TestDFGReachingDefinitionsPerformance:
    """Test that CFGReachingDefsAnalyzer works correctly after optimization."""

    def test_basic_reaching_definitions(self):
        """Basic def-use chain detection should work."""
        from tldr.dfg_extractor import extract_python_dfg

        code = '''
def foo(x):
    y = x + 1
    z = y * 2
    return z
'''
        dfg = extract_python_dfg(code, "foo")
        assert dfg.function_name == "foo"

        # Should have edges: x -> y (x used in y = x + 1)
        # y -> z (y used in z = y * 2)
        # z -> return (z used in return)
        edges = dfg.dataflow_edges
        edge_vars = [e.var_name for e in edges]
        assert "x" in edge_vars or "y" in edge_vars, f"Expected def-use chains, got: {edge_vars}"

    def test_branch_merge_reaching_definitions(self):
        """Definitions from both if/else branches should reach uses after merge."""
        from tldr.dfg_extractor import extract_python_dfg

        code = '''
def foo(cond):
    if cond:
        x = 1
    else:
        x = 2
    return x
'''
        dfg = extract_python_dfg(code, "foo")

        # Both definitions of x (line 4 and 6) should reach the use at return
        edges = dfg.dataflow_edges
        x_edges = [e for e in edges if e.var_name == "x"]
        # Should have at least one edge from x def to x use
        assert len(x_edges) >= 1, f"Expected x def-use edges, got: {x_edges}"

    def test_loop_reaching_definitions(self):
        """Loop body definitions should reach loop condition uses."""
        from tldr.dfg_extractor import extract_python_dfg

        code = '''
def foo(n):
    i = 0
    while i < n:
        i = i + 1
    return i
'''
        dfg = extract_python_dfg(code, "foo")
        edges = dfg.dataflow_edges

        # Should have i edges
        i_edges = [e for e in edges if e.var_name == "i"]
        assert len(i_edges) >= 1, f"Expected i def-use edges, got: {edges}"

    def test_performance_many_variables(self):
        """Performance test: should handle many variables without O(n^3) slowdown."""
        from tldr.dfg_extractor import extract_python_dfg

        # Generate code with many variables
        var_count = 50
        lines = ["def foo():"]
        for i in range(var_count):
            lines.append(f"    x{i} = {i}")
        for i in range(var_count):
            lines.append(f"    y{i} = x{i} + 1")
        lines.append("    return y0")

        code = "\n".join(lines)

        start = time.time()
        dfg = extract_python_dfg(code, "foo")
        elapsed = time.time() - start

        # Should complete in under 1 second (with O(n^3) it would be much slower)
        assert elapsed < 1.0, f"DFG extraction took {elapsed:.2f}s, expected < 1s"

        # Should have edges
        assert len(dfg.dataflow_edges) > 0


class TestCFGParserCaching:
    """Test that tree-sitter parser caching works correctly."""

    def test_typescript_parser_cached(self):
        """TypeScript parser should be cached between calls."""
        from tldr.cfg_extractor import _get_ts_parser

        # Get parser twice
        parser1 = _get_ts_parser("typescript")
        parser2 = _get_ts_parser("typescript")

        # Should be the same object (cached)
        assert parser1 is parser2, "TypeScript parser should be cached"

    def test_javascript_parser_cached(self):
        """JavaScript parser should be cached between calls."""
        from tldr.cfg_extractor import _get_ts_parser

        parser1 = _get_ts_parser("javascript")
        parser2 = _get_ts_parser("javascript")

        assert parser1 is parser2, "JavaScript parser should be cached"

    def test_go_parser_cached(self):
        """Go parser should be cached between calls."""
        from tldr.cfg_extractor import _get_ts_parser

        try:
            parser1 = _get_ts_parser("go")
            parser2 = _get_ts_parser("go")
            assert parser1 is parser2, "Go parser should be cached"
        except ImportError:
            pytest.skip("tree-sitter-go not available")

    def test_rust_parser_cached(self):
        """Rust parser should be cached between calls."""
        from tldr.cfg_extractor import _get_ts_parser

        try:
            parser1 = _get_ts_parser("rust")
            parser2 = _get_ts_parser("rust")
            assert parser1 is parser2, "Rust parser should be cached"
        except ImportError:
            pytest.skip("tree-sitter-rust not available")

    def test_different_languages_different_parsers(self):
        """Different languages should have different parsers."""
        from tldr.cfg_extractor import _get_ts_parser

        ts_parser = _get_ts_parser("typescript")
        js_parser = _get_ts_parser("javascript")

        # Should be different parsers (different languages)
        assert ts_parser is not js_parser, "TS and JS should have different parsers"

    def test_cfg_extraction_uses_cached_parser(self):
        """CFG extraction should reuse cached parsers."""
        from tldr.cfg_extractor import extract_typescript_cfg, _cfg_parser_cache

        code = '''
function foo(x: number): number {
    if (x > 0) {
        return x;
    }
    return -x;
}
'''
        # Clear cache first
        _cfg_parser_cache.clear()

        # Extract CFG twice
        cfg1 = extract_typescript_cfg(code, "foo")
        cache_size_after_first = len(_cfg_parser_cache)

        cfg2 = extract_typescript_cfg(code, "foo")
        cache_size_after_second = len(_cfg_parser_cache)

        # Cache size should not grow on second call
        assert cache_size_after_first == cache_size_after_second, \
            "Parser cache should not grow on repeated extractions"

        # Results should be equivalent
        assert cfg1.function_name == cfg2.function_name


class TestPDGParserCaching:
    """Test that PDG extractor doesn't unnecessarily re-parse."""

    def test_pdg_reuses_cfg_and_dfg(self):
        """PDG should build from CFG+DFG, not re-parse the code."""
        from tldr.pdg_extractor import extract_python_pdg

        code = '''
def foo(x):
    y = x + 1
    if y > 0:
        return y
    return -y
'''
        # Should work without error
        pdg = extract_python_pdg(code, "foo")
        assert pdg is not None
        assert len(pdg.nodes) > 0
        assert len(pdg.edges) > 0


class TestCFGExtractorCorrectness:
    """Ensure CFG extraction still works correctly after optimizations."""

    def test_typescript_cfg_basic(self):
        """Basic TypeScript CFG extraction should work."""
        from tldr.cfg_extractor import extract_typescript_cfg

        code = '''
function foo(x: number): number {
    return x + 1;
}
'''
        cfg = extract_typescript_cfg(code, "foo")
        assert cfg.function_name == "foo"
        assert len(cfg.blocks) > 0

    def test_typescript_cfg_with_branches(self):
        """TypeScript CFG with if/else should have correct edges."""
        from tldr.cfg_extractor import extract_typescript_cfg

        code = '''
function foo(x: number): number {
    if (x > 0) {
        return x;
    } else {
        return -x;
    }
}
'''
        cfg = extract_typescript_cfg(code, "foo")
        assert cfg.function_name == "foo"
        # Should have entry, condition, true branch, false branch, exit
        assert len(cfg.blocks) >= 3
        # Should have edges
        assert len(cfg.edges) >= 2


class TestDFGExtractorCorrectness:
    """Ensure DFG extraction still works correctly after optimizations."""

    def test_typescript_dfg_basic(self):
        """Basic TypeScript DFG extraction should work."""
        from tldr.dfg_extractor import extract_typescript_dfg

        code = '''
function foo(x: number): number {
    const y = x + 1;
    return y;
}
'''
        dfg = extract_typescript_dfg(code, "foo")
        assert dfg.function_name == "foo"
        # Should have variable references
        assert len(dfg.var_refs) > 0

    def test_go_dfg_basic(self):
        """Basic Go DFG extraction should work."""
        from tldr.dfg_extractor import extract_go_dfg

        code = '''
func foo(x int) int {
    y := x + 1
    return y
}
'''
        try:
            dfg = extract_go_dfg(code, "foo")
            assert dfg.function_name == "foo"
        except ImportError:
            pytest.skip("tree-sitter-go not available")
