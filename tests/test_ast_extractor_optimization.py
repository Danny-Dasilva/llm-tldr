"""
Tests for AST extractor single-pass optimization.

These tests verify that:
1. The optimized single-pass NodeVisitor extracts the same info as before
2. ast.walk() is called only once per file (not per function)
3. Performance improves on files with many functions
"""
import ast
import time
import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import patch


class TestSinglePassExtraction:
    """Test that single-pass extraction produces correct results."""

    def test_basic_function_extraction(self):
        """Basic function should be extracted correctly."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
def foo(x: int) -> int:
    """Add one to x."""
    return x + 1
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "foo"
        assert func.return_type == "int"
        assert "x: int" in func.params
        assert func.docstring == "Add one to x."

    def test_class_with_methods_extraction(self):
        """Class with methods should be extracted correctly."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
class MyClass:
    """A test class."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        return self.value

    async def async_method(self) -> str:
        return "async"
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "MyClass"
        assert cls.docstring == "A test class."
        assert len(cls.methods) == 3

        method_names = [m.name for m in cls.methods]
        assert "__init__" in method_names
        assert "get_value" in method_names
        assert "async_method" in method_names

        # Check async flag
        async_method = [m for m in cls.methods if m.name == "async_method"][0]
        assert async_method.is_async is True

    def test_call_graph_extraction(self):
        """Call graph should be correctly extracted."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
def helper():
    return 42

def caller():
    x = helper()
    return x

def another():
    caller()
    helper()
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        cg = result.call_graph

        # caller -> helper
        assert "caller" in cg.calls
        assert "helper" in cg.calls["caller"]

        # another -> caller, helper
        assert "another" in cg.calls
        assert "caller" in cg.calls["another"]
        assert "helper" in cg.calls["another"]

        # Reverse: helper called by caller and another
        assert "helper" in cg.called_by
        assert "caller" in cg.called_by["helper"]
        assert "another" in cg.called_by["helper"]

    def test_nested_function_extraction(self):
        """Nested functions should be extracted."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
def outer():
    def inner():
        return 1
    return inner()
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        func_names = [f.name for f in result.functions]
        assert "outer" in func_names
        assert "inner" in func_names

    def test_imports_extraction(self):
        """Import statements should be extracted."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
import os
import sys
from pathlib import Path
from typing import List, Dict

def foo():
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        assert len(result.imports) == 4

        # Check regular imports
        regular_imports = [i for i in result.imports if not i.is_from]
        assert len(regular_imports) == 2

        # Check from imports
        from_imports = [i for i in result.imports if i.is_from]
        assert len(from_imports) == 2

        typing_import = [i for i in from_imports if i.module == "typing"][0]
        assert "List" in typing_import.names
        assert "Dict" in typing_import.names

    def test_decorators_extraction(self):
        """Decorators should be extracted."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
def decorator(f):
    return f

@decorator
@staticmethod
def decorated():
    pass

class MyClass:
    @classmethod
    def class_method(cls):
        pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        decorated_func = [f for f in result.functions if f.name == "decorated"][0]
        assert "decorator" in decorated_func.decorators
        assert "staticmethod" in decorated_func.decorators

        cls = result.classes[0]
        class_method = [m for m in cls.methods if m.name == "class_method"][0]
        assert "classmethod" in class_method.decorators


class TestSinglePassPerformance:
    """Test that optimization reduces ast.walk() calls."""

    def test_walk_called_once_per_file(self):
        """ast.walk should be called at most twice per file (not per function)."""
        from tldr.ast_extractor import PythonASTExtractor
        import tempfile

        # Generate code with many functions
        lines = []
        for i in range(20):
            lines.append(f'''
def func_{i}(x):
    y = x + 1
    return y
''')
        code = "\n".join(lines)

        walk_call_count = 0
        original_walk = ast.walk

        def counting_walk(node):
            nonlocal walk_call_count
            walk_call_count += 1
            return original_walk(node)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            with patch('ast.walk', side_effect=counting_walk):
                # Also need to patch in the module
                import tldr.ast_extractor as ast_module
                original_module_walk = ast_module.ast.walk
                ast_module.ast.walk = counting_walk
                try:
                    extractor = PythonASTExtractor()
                    result = extractor.extract(f.name)
                finally:
                    ast_module.ast.walk = original_module_walk

        # With 20 functions, old approach would call walk 20+ times
        # New approach should call walk at most 2 times (defined_names + full pass)
        # Allow some margin for implementation details
        assert walk_call_count <= 3, \
            f"ast.walk called {walk_call_count} times, expected <= 3 for single-pass"

        # Verify extraction still works
        assert len(result.functions) == 20

    def test_performance_many_functions(self):
        """Extraction should be fast with many functions."""
        from tldr.ast_extractor import extract_python
        import tempfile

        # Generate code with 100 functions
        lines = []
        for i in range(100):
            lines.append(f'''
def func_{i}(x, y, z):
    """Function {i} docstring."""
    a = x + y
    b = a * z
    c = func_{(i + 1) % 100}(a, b, z)
    return c
''')
        code = "\n".join(lines)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            start = time.time()
            result = extract_python(f.name)
            elapsed = time.time() - start

        # Should complete quickly (under 0.5 seconds for 100 functions)
        assert elapsed < 0.5, f"Extraction took {elapsed:.2f}s, expected < 0.5s"

        # Verify correctness
        assert len(result.functions) == 100
        # Should have call graph entries
        assert len(result.call_graph.calls) > 0

    def test_performance_deeply_nested_classes(self):
        """Should handle deeply nested classes efficiently."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
class Outer:
    def outer_method(self):
        pass

    class Middle:
        def middle_method(self):
            pass

        class Inner:
            def inner_method(self):
                pass

            def another_inner(self):
                self.inner_method()
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            start = time.time()
            result = extract_python(f.name)
            elapsed = time.time() - start

        # Should be fast
        assert elapsed < 0.1, f"Extraction took {elapsed:.2f}s"

        # Should find all classes
        class_names = [c.name for c in result.classes]
        assert "Outer" in class_names
        assert "Middle" in class_names
        assert "Inner" in class_names


class TestMethodCallGraphInClass:
    """Test call graph extraction within class methods."""

    def test_method_to_method_calls(self):
        """Method-to-method calls within a class should be tracked."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        result = 0
        for _ in range(b):
            result = self.add(result, a)
        return result
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        cg = result.call_graph

        # multiply calls add
        # The caller name should be qualified: Calculator.multiply
        caller_with_add = [c for c in cg.calls if "add" in cg.calls.get(c, [])]
        assert len(caller_with_add) >= 1, f"Expected multiply to call add, got: {cg.calls}"

    def test_function_calls_method(self):
        """Top-level function calling a method should be tracked."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
def helper():
    return 1

class MyClass:
    def use_helper(self):
        return helper()
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        cg = result.call_graph

        # MyClass.use_helper calls helper
        assert "helper" in cg.called_by, f"Expected helper to be called, got: {cg.called_by}"


class TestEdgeCases:
    """Test edge cases in extraction."""

    def test_empty_file(self):
        """Empty file should not crash."""
        from tldr.ast_extractor import extract_python
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")
            f.flush()
            result = extract_python(f.name)

        assert result is not None
        assert len(result.functions) == 0
        assert len(result.classes) == 0

    def test_syntax_error_file(self):
        """File with syntax error should return partial info."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
def valid_func():
    pass

def invalid(
    # missing closing paren
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        # Should return a result (may be empty due to parse error)
        assert result is not None

    def test_lambda_not_extracted_as_function(self):
        """Lambda expressions should not appear as top-level functions."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
add = lambda x, y: x + y

def real_func():
    double = lambda x: x * 2
    return double(5)
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        # Only real_func should be in functions
        func_names = [f.name for f in result.functions]
        assert "real_func" in func_names
        # Lambdas are anonymous, so no "add" or "double" functions
        assert "add" not in func_names
        assert "double" not in func_names

    def test_comprehension_calls_not_function_defs(self):
        """Calls inside comprehensions should be tracked, not as new functions."""
        from tldr.ast_extractor import extract_python
        import tempfile

        code = '''
def process(x):
    return x * 2

def main():
    data = [1, 2, 3]
    result = [process(x) for x in data]
    return result
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extract_python(f.name)

        # Should have main -> process in call graph
        cg = result.call_graph
        assert "main" in cg.calls
        assert "process" in cg.calls["main"]
