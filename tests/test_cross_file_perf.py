"""Performance tests for cross_file_calls.py optimizations."""

import pytest
from pathlib import Path
import tempfile


class TestParserCaching:
    """Test that tree-sitter parsers are cached properly."""

    def test_ts_parser_cached(self):
        """TypeScript parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_ts_parser, TREE_SITTER_AVAILABLE
        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter-typescript not available")

        parser1 = _get_ts_parser()
        parser2 = _get_ts_parser()
        # Should be the same parser instance (cached)
        assert parser1 is parser2

    def test_go_parser_cached(self):
        """Go parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_go_parser, TREE_SITTER_GO_AVAILABLE
        if not TREE_SITTER_GO_AVAILABLE:
            pytest.skip("tree-sitter-go not available")

        parser1 = _get_go_parser()
        parser2 = _get_go_parser()
        assert parser1 is parser2

    def test_rust_parser_cached(self):
        """Rust parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_rust_parser, TREE_SITTER_RUST_AVAILABLE
        if not TREE_SITTER_RUST_AVAILABLE:
            pytest.skip("tree-sitter-rust not available")

        parser1 = _get_rust_parser()
        parser2 = _get_rust_parser()
        assert parser1 is parser2

    def test_java_parser_cached(self):
        """Java parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_java_parser, TREE_SITTER_JAVA_AVAILABLE
        if not TREE_SITTER_JAVA_AVAILABLE:
            pytest.skip("tree-sitter-java not available")

        parser1 = _get_java_parser()
        parser2 = _get_java_parser()
        assert parser1 is parser2

    def test_c_parser_cached(self):
        """C parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_c_parser, TREE_SITTER_C_AVAILABLE
        if not TREE_SITTER_C_AVAILABLE:
            pytest.skip("tree-sitter-c not available")

        parser1 = _get_c_parser()
        parser2 = _get_c_parser()
        assert parser1 is parser2

    def test_ruby_parser_cached(self):
        """Ruby parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_ruby_parser, TREE_SITTER_RUBY_AVAILABLE
        if not TREE_SITTER_RUBY_AVAILABLE:
            pytest.skip("tree-sitter-ruby not available")

        parser1 = _get_ruby_parser()
        parser2 = _get_ruby_parser()
        assert parser1 is parser2

    def test_php_parser_cached(self):
        """PHP parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_php_parser, TREE_SITTER_PHP_AVAILABLE
        if not TREE_SITTER_PHP_AVAILABLE:
            pytest.skip("tree-sitter-php not available")

        parser1 = _get_php_parser()
        parser2 = _get_php_parser()
        assert parser1 is parser2

    def test_cpp_parser_cached(self):
        """C++ parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_cpp_parser, TREE_SITTER_CPP_AVAILABLE
        if not TREE_SITTER_CPP_AVAILABLE:
            pytest.skip("tree-sitter-cpp not available")

        parser1 = _get_cpp_parser()
        parser2 = _get_cpp_parser()
        assert parser1 is parser2

    def test_kotlin_parser_cached(self):
        """Kotlin parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_kotlin_parser, TREE_SITTER_KOTLIN_AVAILABLE
        if not TREE_SITTER_KOTLIN_AVAILABLE:
            pytest.skip("tree-sitter-kotlin not available")

        parser1 = _get_kotlin_parser()
        parser2 = _get_kotlin_parser()
        assert parser1 is parser2

    def test_swift_parser_cached(self):
        """Swift parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_swift_parser, TREE_SITTER_SWIFT_AVAILABLE
        if not TREE_SITTER_SWIFT_AVAILABLE:
            pytest.skip("tree-sitter-swift not available")

        parser1 = _get_swift_parser()
        parser2 = _get_swift_parser()
        assert parser1 is parser2

    def test_csharp_parser_cached(self):
        """C# parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_csharp_parser, TREE_SITTER_CSHARP_AVAILABLE
        if not TREE_SITTER_CSHARP_AVAILABLE:
            pytest.skip("tree-sitter-c-sharp not available")

        parser1 = _get_csharp_parser()
        parser2 = _get_csharp_parser()
        assert parser1 is parser2

    def test_scala_parser_cached(self):
        """Scala parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_scala_parser, TREE_SITTER_SCALA_AVAILABLE
        if not TREE_SITTER_SCALA_AVAILABLE:
            pytest.skip("tree-sitter-scala not available")

        parser1 = _get_scala_parser()
        parser2 = _get_scala_parser()
        assert parser1 is parser2

    def test_lua_parser_cached(self):
        """Lua parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_lua_parser, TREE_SITTER_LUA_AVAILABLE
        if not TREE_SITTER_LUA_AVAILABLE:
            pytest.skip("tree-sitter-lua not available")

        parser1 = _get_lua_parser()
        parser2 = _get_lua_parser()
        assert parser1 is parser2

    def test_luau_parser_cached(self):
        """Luau parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_luau_parser, TREE_SITTER_LUAU_AVAILABLE
        if not TREE_SITTER_LUAU_AVAILABLE:
            pytest.skip("tree-sitter-luau not available")

        parser1 = _get_luau_parser()
        parser2 = _get_luau_parser()
        assert parser1 is parser2

    def test_elixir_parser_cached(self):
        """Elixir parser should be cached, returning same instance."""
        from tldr.cross_file_calls import _get_elixir_parser, TREE_SITTER_ELIXIR_AVAILABLE
        if not TREE_SITTER_ELIXIR_AVAILABLE:
            pytest.skip("tree-sitter-elixir not available")

        parser1 = _get_elixir_parser()
        parser2 = _get_elixir_parser()
        assert parser1 is parser2


class TestProjectCallGraphSet:
    """Test that ProjectCallGraph uses set for edges (O(1) dedup)."""

    def test_edges_uses_set(self):
        """ProjectCallGraph._edges should be a set for O(1) deduplication."""
        from tldr.cross_file_calls import ProjectCallGraph

        graph = ProjectCallGraph()
        # The internal _edges should be a set
        assert isinstance(graph._edges, set)

    def test_add_edge_deduplicates(self):
        """Adding the same edge twice should not duplicate."""
        from tldr.cross_file_calls import ProjectCallGraph

        graph = ProjectCallGraph()
        graph.add_edge("a.py", "foo", "b.py", "bar")
        graph.add_edge("a.py", "foo", "b.py", "bar")  # duplicate

        # Should only have one edge
        assert len(graph.edges) == 1

    def test_edges_property_returns_set(self):
        """The edges property should return a set."""
        from tldr.cross_file_calls import ProjectCallGraph

        graph = ProjectCallGraph()
        graph.add_edge("a.py", "foo", "b.py", "bar")
        assert isinstance(graph.edges, set)


class TestGoCallGraphLookup:
    """Test that Go call graph uses efficient lookup instead of O(n^2)."""

    def test_func_index_lookup_efficient(self):
        """Go call graph should use dict lookup, not iterate over all keys."""
        # This test verifies the pattern is correct by checking the implementation
        # doesn't use nested iteration over func_index items
        import inspect
        from tldr.cross_file_calls import _build_go_call_graph

        source = inspect.getsource(_build_go_call_graph)

        # The O(n^2) pattern was: for key, file_path in func_index.items():
        # This should NOT be present in the efficient implementation
        # Instead, there should be direct dict lookups like func_index.get() or func_index[key]

        # Check that we don't iterate over func_index.items() inside a loop
        lines = source.split('\n')
        in_nested_loop = False
        for i, line in enumerate(lines):
            # Detect we're in a for loop over calls
            if 'for call_type, call_target in calls' in line:
                in_nested_loop = True
            # If we find iteration over func_index while in nested loop, fail
            if in_nested_loop and 'for key, file_path in func_index.items()' in line:
                pytest.fail(
                    f"Found O(n^2) pattern: iterating over func_index.items() "
                    f"inside call loop at line {i}: {line.strip()}"
                )
            # Reset when we exit the nested block (dedent)
            if in_nested_loop and line and not line.startswith(' ' * 16) and line.strip():
                if 'for caller_func' in line or 'def ' in line:
                    in_nested_loop = False
