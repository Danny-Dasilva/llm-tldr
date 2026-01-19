"""Tests for data structure optimizations.

Tests for:
1. CallVisitor using O(1) set lookups instead of O(n) list lookups
2. Dataclasses with __slots__ for memory efficiency
"""

import ast
import pytest
import sys


class TestCallVisitorEfficiency:
    """Test that CallVisitor uses efficient data structures."""

    def test_callvisitor_has_shadow_sets(self):
        """CallVisitor should have shadow sets for O(1) lookups."""
        from tldr.cross_file_calls import CallVisitor

        visitor = CallVisitor()
        # Should have shadow sets for O(1) membership checks
        assert hasattr(visitor, '_calls_set'), "CallVisitor should have _calls_set for O(1) lookup"
        assert hasattr(visitor, '_refs_set'), "CallVisitor should have _refs_set for O(1) lookup"
        assert isinstance(visitor._calls_set, set)
        assert isinstance(visitor._refs_set, set)

    def test_callvisitor_shadow_sets_sync_with_lists(self):
        """Shadow sets should stay in sync with lists during visitation."""
        from tldr.cross_file_calls import CallVisitor

        code = '''
def foo():
    bar()
    baz()
    bar()  # duplicate call
    qux()
'''
        tree = ast.parse(code)
        func_node = tree.body[0]  # foo function

        visitor = CallVisitor(defined_funcs={'bar', 'baz', 'qux'})
        visitor.visit(func_node)

        # Lists and sets should have same elements
        assert set(visitor.calls) == visitor._calls_set, "calls list and _calls_set should have same elements"
        assert set(visitor.refs) == visitor._refs_set, "refs list and _refs_set should have same elements"

    def test_callvisitor_no_duplicate_calls(self):
        """CallVisitor should not add duplicate calls to the list."""
        from tldr.cross_file_calls import CallVisitor

        code = '''
def foo():
    bar()
    bar()
    bar()
'''
        tree = ast.parse(code)
        func_node = tree.body[0]

        visitor = CallVisitor(defined_funcs={'bar'})
        visitor.visit(func_node)

        # Should only have one 'bar' in calls list (no duplicates)
        assert visitor.calls.count('bar') == 1, f"Expected 1 'bar' but got {visitor.calls.count('bar')}: {visitor.calls}"

    def test_callvisitor_no_duplicate_refs(self):
        """CallVisitor should not add duplicate refs to the list."""
        from tldr.cross_file_calls import CallVisitor

        code = '''
def foo():
    handlers = [process, process, process]
'''
        tree = ast.parse(code)
        func_node = tree.body[0]

        visitor = CallVisitor(defined_funcs={'process'})
        visitor.visit(func_node)

        # Should only have one 'process' in refs list (no duplicates)
        assert visitor.refs.count('process') == 1, f"Expected 1 'process' but got {visitor.refs.count('process')}: {visitor.refs}"


class TestDataclassSlots:
    """Test that dataclasses use __slots__ for memory efficiency."""

    def test_cfgblock_has_slots(self):
        """CFGBlock should have __slots__ defined."""
        from tldr.cfg_extractor import CFGBlock

        # Check for __slots__ - with slots=True, instances won't have __dict__
        block = CFGBlock(id=1, start_line=1, end_line=10, block_type="body")
        assert not hasattr(block, '__dict__'), "CFGBlock should use __slots__ (no __dict__)"

    def test_cfgedge_has_slots(self):
        """CFGEdge should have __slots__ defined."""
        from tldr.cfg_extractor import CFGEdge

        edge = CFGEdge(source_id=1, target_id=2, edge_type="unconditional")
        assert not hasattr(edge, '__dict__'), "CFGEdge should use __slots__ (no __dict__)"

    def test_varref_has_slots(self):
        """VarRef should have __slots__ defined."""
        from tldr.dfg_extractor import VarRef

        ref = VarRef(name="x", ref_type="definition", line=1, column=0)
        assert not hasattr(ref, '__dict__'), "VarRef should use __slots__ (no __dict__)"

    def test_dataflowedge_has_slots(self):
        """DataflowEdge should have __slots__ defined."""
        from tldr.dfg_extractor import VarRef, DataflowEdge

        def_ref = VarRef(name="x", ref_type="definition", line=1, column=0)
        use_ref = VarRef(name="x", ref_type="use", line=5, column=0)
        edge = DataflowEdge(def_ref=def_ref, use_ref=use_ref)
        assert not hasattr(edge, '__dict__'), "DataflowEdge should use __slots__ (no __dict__)"

    def test_projectcallgraph_has_slots(self):
        """ProjectCallGraph should have __slots__ defined."""
        from tldr.cross_file_calls import ProjectCallGraph

        graph = ProjectCallGraph()
        assert not hasattr(graph, '__dict__'), "ProjectCallGraph should use __slots__ (no __dict__)"

    def test_slotted_dataclasses_still_functional(self):
        """Slotted dataclasses should still work correctly."""
        from tldr.cfg_extractor import CFGBlock, CFGEdge
        from tldr.dfg_extractor import VarRef, DataflowEdge
        from tldr.cross_file_calls import ProjectCallGraph

        # Test CFGBlock
        block = CFGBlock(id=1, start_line=1, end_line=10, block_type="body")
        assert block.id == 1
        assert block.to_dict()["id"] == 1

        # Test CFGEdge
        edge = CFGEdge(source_id=1, target_id=2, edge_type="true", condition="x > 0")
        assert edge.source_id == 1
        assert edge.to_dict()["condition"] == "x > 0"

        # Test VarRef
        ref = VarRef(name="x", ref_type="definition", line=1, column=0)
        assert ref.name == "x"
        assert ref.to_dict()["type"] == "definition"

        # Test DataflowEdge
        def_ref = VarRef(name="x", ref_type="definition", line=1, column=0)
        use_ref = VarRef(name="x", ref_type="use", line=5, column=4)
        df_edge = DataflowEdge(def_ref=def_ref, use_ref=use_ref)
        assert df_edge.var_name == "x"
        assert df_edge.to_dict()["def_line"] == 1

        # Test ProjectCallGraph
        graph = ProjectCallGraph()
        graph.add_edge("a.py", "foo", "b.py", "bar")
        assert ("a.py", "foo", "b.py", "bar") in graph


class TestMemoryEfficiency:
    """Test that slotted classes use less memory."""

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="sys.getsizeof behavior varies")
    def test_slotted_dataclasses_smaller(self):
        """Slotted dataclasses should use less memory than unslotted ones."""
        import sys
        from tldr.cfg_extractor import CFGBlock

        block = CFGBlock(id=1, start_line=1, end_line=10, block_type="body")

        # With __slots__, objects shouldn't have __dict__
        # This is a basic sanity check - actual memory savings depend on Python version
        assert not hasattr(block, '__dict__'), "Slotted class should not have __dict__"
