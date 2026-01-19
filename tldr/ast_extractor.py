#!/usr/bin/env python3
"""
AST-based code structure extractor with full metadata.

Extracts:
- Function signatures WITH return types
- Class hierarchy (inheritance)
- Import dependencies
- Docstrings

Supports:
- Python (via ast module) - full support
- TypeScript/JavaScript (via tree-sitter) - planned
- Other languages (via tree-sitter) - planned
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Extracted function/method information."""
    name: str
    params: list[str]
    return_type: str | None
    docstring: str | None
    is_method: bool = False
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    line_number: int = 0

    def signature(self) -> str:
        """Return full signature string."""
        async_prefix = "async " if self.is_async else ""
        params_str = ", ".join(self.params)
        ret = f" -> {self.return_type}" if self.return_type else ""
        return f"{async_prefix}def {self.name}({params_str}){ret}"


@dataclass
class ClassInfo:
    """Extracted class information."""
    name: str
    bases: list[str]
    docstring: str | None
    methods: list[FunctionInfo] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    line_number: int = 0

    def signature(self) -> str:
        """Return class definition signature."""
        bases_str = ", ".join(self.bases) if self.bases else ""
        return f"class {self.name}({bases_str})" if bases_str else f"class {self.name}"


@dataclass
class ImportInfo:
    """Extracted import information."""
    module: str
    names: list[str]  # Empty for 'import x', filled for 'from x import y, z'
    is_from: bool = False
    line_number: int = 0

    def statement(self) -> str:
        """Return import statement string."""
        if self.is_from:
            names_str = ", ".join(self.names)
            return f"from {self.module} import {names_str}"
        return f"import {self.module}"


@dataclass
class CallGraphInfo:
    """Call graph showing function relationships."""
    calls: dict[str, list[str]] = field(default_factory=dict)  # func -> [called funcs]
    called_by: dict[str, list[str]] = field(default_factory=dict)  # func -> [callers]

    def add_call(self, caller: str, callee: str):
        """Record a function call."""
        if caller not in self.calls:
            self.calls[caller] = []
        if callee not in self.calls[caller]:
            self.calls[caller].append(callee)

        if callee not in self.called_by:
            self.called_by[callee] = []
        if caller not in self.called_by[callee]:
            self.called_by[callee].append(caller)


@dataclass
class ModuleInfo:
    """Complete module extraction result."""
    file_path: str
    language: str
    docstring: str | None
    imports: list[ImportInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    call_graph: CallGraphInfo = field(default_factory=CallGraphInfo)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "docstring": self.docstring,
            "imports": [
                {"module": i.module, "names": i.names, "is_from": i.is_from}
                for i in self.imports
            ],
            "classes": [
                {
                    "name": c.name,
                    "line_number": c.line_number,
                    "signature": c.signature(),
                    "bases": c.bases,
                    "docstring": c.docstring,
                    "decorators": c.decorators,
                    "methods": [
                        {
                            "name": m.name,
                            "line_number": m.line_number,
                            "signature": m.signature(),
                            "params": m.params,
                            "return_type": m.return_type,
                            "docstring": m.docstring,
                            "is_async": m.is_async,
                            "decorators": m.decorators,
                        }
                        for m in c.methods
                    ],
                }
                for c in self.classes
            ],
            "functions": [
                {
                    "name": f.name,
                    "line_number": f.line_number,
                    "signature": f.signature(),
                    "params": f.params,
                    "return_type": f.return_type,
                    "docstring": f.docstring,
                    "is_async": f.is_async,
                    "decorators": f.decorators,
                }
                for f in self.functions
            ],
            "call_graph": {
                "calls": self.call_graph.calls,
                "called_by": self.call_graph.called_by,
            } if self.call_graph.calls else {},
        }

    def to_compact(self) -> dict[str, Any]:
        """Compact format optimized for LLM context."""
        result: dict[str, Any] = {
            "file": Path(self.file_path).name,
            "lang": self.language,
        }

        if self.docstring:
            # Truncate long docstrings
            doc = self.docstring[:200] + "..." if len(self.docstring) > 200 else self.docstring
            result["doc"] = doc

        if self.imports:
            result["imports"] = [i.statement() for i in self.imports]

        if self.classes:
            result["classes"] = {}
            for c in self.classes:
                class_info: dict[str, Any] = {"bases": c.bases} if c.bases else {}
                if c.docstring:
                    class_info["doc"] = c.docstring[:100] + "..." if len(c.docstring) > 100 else c.docstring
                if c.methods:
                    class_info["methods"] = [m.signature() for m in c.methods]
                result["classes"][c.name] = class_info

        if self.functions:
            result["functions"] = [f.signature() for f in self.functions]

        if self.call_graph.calls:
            result["calls"] = self.call_graph.calls

        return result


# Standalone helper functions for single-pass extraction
# These are module-level to avoid method lookup overhead in hot paths

def _node_to_str(node: ast.AST | None) -> str:
    """Convert AST node to string representation."""
    if node is None:
        return ""
    # Python 3.9+ has ast.unparse
    try:
        return ast.unparse(node)
    except AttributeError:
        # Fallback for older Python
        return _manual_unparse(node)


def _manual_unparse(node: ast.AST) -> str:
    """Manual unparse for older Python versions."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_manual_unparse(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        return f"{_manual_unparse(node.value)}[{_manual_unparse(node.slice)}]"
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Tuple):
        elts = ", ".join(_manual_unparse(e) for e in node.elts)
        return f"({elts})"
    elif isinstance(node, ast.List):
        elts = ", ".join(_manual_unparse(e) for e in node.elts)
        return f"[{elts}]"
    elif isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.BitOr):
            return f"{_manual_unparse(node.left)} | {_manual_unparse(node.right)}"
    elif isinstance(node, ast.Call):
        func = _manual_unparse(node.func)
        args = ", ".join(_manual_unparse(a) for a in node.args)
        return f"{func}({args})"
    return "<unknown>"


def _format_arg(arg: ast.arg) -> str:
    """Format a single argument with optional type annotation."""
    if arg.annotation:
        return f"{arg.arg}: {_node_to_str(arg.annotation)}"
    return arg.arg


def _extract_params(args: ast.arguments) -> list[str]:
    """Extract parameter list with type annotations."""
    params = []

    # Positional-only params (before /)
    for arg in args.posonlyargs:
        params.append(_format_arg(arg))

    if args.posonlyargs:
        params.append("/")

    # Regular positional/keyword params
    defaults_start = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        param = _format_arg(arg)
        # Add default value indicator
        if i >= defaults_start:
            default_idx = i - defaults_start
            default = args.defaults[default_idx]
            param += f" = {_node_to_str(default)}"
        params.append(param)

    # *args
    if args.vararg:
        params.append(f"*{_format_arg(args.vararg)}")
    elif args.kwonlyargs:
        params.append("*")

    # Keyword-only params
    kw_defaults_map = {i: d for i, d in enumerate(args.kw_defaults) if d is not None}
    for i, arg in enumerate(args.kwonlyargs):
        param = _format_arg(arg)
        if i in kw_defaults_map:
            param += f" = {_node_to_str(kw_defaults_map[i])}"
        params.append(param)

    # **kwargs
    if args.kwarg:
        params.append(f"**{_format_arg(args.kwarg)}")

    return params


def _extract_function_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    is_method: bool = False
) -> FunctionInfo:
    """Extract function/method information."""
    params = _extract_params(node.args)
    return_type = _node_to_str(node.returns) if node.returns else None
    decorators = [_node_to_str(d) for d in node.decorator_list]

    return FunctionInfo(
        name=node.name,
        params=params,
        return_type=return_type,
        docstring=ast.get_docstring(node),
        is_method=is_method,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        decorators=decorators,
        line_number=node.lineno,
    )


class _SinglePassVisitor(ast.NodeVisitor):
    """
    Single-pass AST visitor that collects all extraction info in one traversal.

    This replaces the previous O(n*m) approach of calling ast.walk() for each
    function to extract calls. Now we do ONE traversal and track context.
    """

    def __init__(self):
        # Collected data
        self.defined_names: set[str] = set()
        self.imports: list[ImportInfo] = []
        self.classes: list[ClassInfo] = []
        self.functions: list[FunctionInfo] = []
        self.call_graph = CallGraphInfo()

        # Context tracking during traversal
        self._class_stack: list[str] = []  # Stack of class names for nesting
        self._func_stack: list[str] = []   # Stack of function names for nesting
        self._current_caller: str | None = None  # Current function/method for call tracking

        # Deferred nested function processing
        self._nested_functions: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]] = []

    def _get_qualified_name(self, name: str) -> str:
        """Get qualified name including class path."""
        if self._class_stack:
            return f"{'.'.join(self._class_stack)}.{name}"
        return name

    def visit_Import(self, node: ast.Import):
        """Handle import statements."""
        for alias in node.names:
            self.imports.append(ImportInfo(
                module=alias.name,
                names=[],
                is_from=False,
                line_number=node.lineno,
            ))
        # Don't visit children of import

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle from...import statements."""
        module_name = node.module or ""
        names = [alias.name for alias in node.names]
        self.imports.append(ImportInfo(
            module=module_name,
            names=names,
            is_from=True,
            line_number=node.lineno,
        ))
        # Don't visit children of import

    def visit_ClassDef(self, node: ast.ClassDef):
        """Handle class definitions."""
        bases = [_node_to_str(base) for base in node.bases]
        decorators = [_node_to_str(d) for d in node.decorator_list]

        class_info = ClassInfo(
            name=node.name,
            bases=bases,
            docstring=ast.get_docstring(node),
            decorators=decorators,
            line_number=node.lineno,
        )

        # Track class context for nested items
        self._class_stack.append(node.name)

        # Process class body for methods and nested classes
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Record method name for call graph filtering
                self.defined_names.add(item.name)

                # Extract method
                method = _extract_function_info(item, is_method=True)
                class_info.methods.append(method)

                # Track method for call graph
                caller_name = self._get_qualified_name(item.name)
                old_caller = self._current_caller
                self._current_caller = caller_name

                # Visit method body for calls
                self._visit_body_for_calls(item.body)

                self._current_caller = old_caller

            elif isinstance(item, ast.ClassDef):
                # Nested class - recurse
                self.visit_ClassDef(item)

        self._class_stack.pop()
        self.classes.append(class_info)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definitions."""
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle async function definitions."""
        self._handle_function(node)

    def _handle_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        """Common handler for function/async function definitions."""
        # Only process top-level functions here (not nested)
        # Nested functions are handled when we visit their parent
        if self._func_stack:
            # This is a nested function - record it but don't add to functions yet
            parent_name = self._func_stack[-1]
            self._nested_functions.append((node, parent_name))
            return

        # Record defined name
        self.defined_names.add(node.name)

        # Extract function info
        func_info = _extract_function_info(node, is_method=False)
        self.functions.append(func_info)

        # Track context for call extraction
        self._func_stack.append(node.name)
        old_caller = self._current_caller
        self._current_caller = node.name

        # Visit body for calls and nested functions
        self._visit_body_for_calls_and_nested(node.body, node.name)

        self._current_caller = old_caller
        self._func_stack.pop()

    def _visit_body_for_calls(self, body: list[ast.stmt]):
        """Visit a function/method body to extract calls only."""
        for stmt in body:
            self._extract_calls_from_node(stmt)

    def _visit_body_for_calls_and_nested(self, body: list[ast.stmt], parent_name: str):
        """Visit a function body to extract calls and nested functions."""
        for stmt in body:
            # Check for nested function definitions
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Record defined name
                self.defined_names.add(stmt.name)

                # Extract nested function
                func_info = _extract_function_info(stmt, is_method=False)
                func_info.decorators.insert(0, f"nested_in:{parent_name}")
                self.functions.append(func_info)

                # Track calls from nested function
                old_caller = self._current_caller
                self._current_caller = stmt.name
                self._visit_body_for_calls_and_nested(stmt.body, stmt.name)
                self._current_caller = old_caller
            else:
                # Extract calls from this statement
                self._extract_calls_from_node(stmt)

    def _extract_calls_from_node(self, node: ast.AST):
        """Extract all calls from an AST node (recursively)."""
        if self._current_caller is None:
            return

        # Use iter_child_nodes to avoid creating new walk generators
        nodes_to_visit = [node]
        while nodes_to_visit:
            current = nodes_to_visit.pop()

            if isinstance(current, ast.Call):
                callee = self._get_call_name(current)
                if callee:
                    # We'll filter by defined_names later
                    # For now, record all calls
                    self.call_graph.add_call(self._current_caller, callee)

            # Skip nested function definitions - they have their own caller context
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Add children to visit
            nodes_to_visit.extend(ast.iter_child_nodes(current))

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Get the name of a called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # For method calls like self.method() or obj.method()
            return node.func.attr
        return None

    def finalize(self, defined_names: set[str]):
        """
        Finalize extraction by filtering call graph to only defined names.

        This is called after the first pass to filter external calls.
        """
        # Filter call graph to only include calls to defined functions
        filtered_calls: dict[str, list[str]] = {}
        for caller, callees in self.call_graph.calls.items():
            filtered = [c for c in callees if c in defined_names]
            if filtered:
                filtered_calls[caller] = filtered

        # Rebuild call graph with filtered data
        self.call_graph = CallGraphInfo()
        for caller, callees in filtered_calls.items():
            for callee in callees:
                self.call_graph.add_call(caller, callee)


class PythonASTExtractor:
    """Extract code structure from Python files using AST."""

    def extract(self, file_path: str | Path) -> ModuleInfo:
        """Extract module information from a Python file."""
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return ModuleInfo(
                file_path=str(file_path),
                language="python",
                docstring=None,
            )

        # Single-pass extraction using NodeVisitor
        visitor = _SinglePassVisitor()

        # Visit top-level nodes
        for node in ast.iter_child_nodes(tree):
            visitor.visit(node)

        # Finalize: filter call graph to only defined functions
        visitor.finalize(visitor.defined_names)

        # Build module info from visitor results
        module_info = ModuleInfo(
            file_path=str(file_path),
            language="python",
            docstring=ast.get_docstring(tree),
            imports=visitor.imports,
            classes=visitor.classes,
            functions=visitor.functions,
            call_graph=visitor.call_graph,
        )

        return module_info

    def _extract_class(
        self,
        node: ast.ClassDef,
        call_graph: CallGraphInfo | None = None,
        defined_names: set[str] | None = None,
        module_info: ModuleInfo | None = None,
        parent_path: str = "",
    ) -> ClassInfo:
        """Extract class information, including nested classes."""
        bases = []
        for base in node.bases:
            bases.append(self._node_to_str(base))

        decorators = [self._node_to_str(d) for d in node.decorator_list]

        # Build qualified name for nested class tracking
        qualified_name = f"{parent_path}.{node.name}" if parent_path else node.name

        class_info = ClassInfo(
            name=node.name,
            bases=bases,
            docstring=ast.get_docstring(node),
            decorators=decorators,
            line_number=node.lineno,
        )

        # Extract methods and nested classes
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._extract_function(item, is_method=True)
                class_info.methods.append(method)
                # Extract calls from this method
                if call_graph and defined_names:
                    caller_name = f"{qualified_name}.{item.name}"
                    self._extract_calls(item, caller_name, call_graph, defined_names)

            elif isinstance(item, ast.ClassDef):
                # Recursively extract nested classes
                nested_class = self._extract_class(
                    item,
                    call_graph=call_graph,
                    defined_names=defined_names,
                    module_info=module_info,
                    parent_path=qualified_name,
                )
                # Add nested class to module's classes list
                if module_info is not None:
                    module_info.classes.append(nested_class)
                # Also add nested class methods to module's functions list for discoverability
                for method in nested_class.methods:
                    if module_info is not None:
                        # Create a copy with qualified name
                        nested_method = FunctionInfo(
                            name=method.name,
                            params=method.params,
                            return_type=method.return_type,
                            docstring=method.docstring,
                            is_method=True,
                            is_async=method.is_async,
                            decorators=[f"nested_in:{qualified_name}.{nested_class.name}"] + method.decorators,
                            line_number=method.line_number,
                        )
                        module_info.functions.append(nested_method)

        return class_info

    def _extract_calls(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        caller_name: str,
        call_graph: CallGraphInfo,
        defined_names: set[str],
    ):
        """Extract function calls from a function body."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = self._get_call_name(child)
                if callee and callee in defined_names:
                    call_graph.add_call(caller_name, callee)

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Get the name of a called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # For method calls like self.method() or obj.method()
            # We only track the method name for simplicity
            return node.func.attr
        return None

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_method: bool = False
    ) -> FunctionInfo:
        """Extract function/method information."""
        params = self._extract_params(node.args)
        return_type = self._node_to_str(node.returns) if node.returns else None
        decorators = [self._node_to_str(d) for d in node.decorator_list]

        return FunctionInfo(
            name=node.name,
            params=params,
            return_type=return_type,
            docstring=ast.get_docstring(node),
            is_method=is_method,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators,
            line_number=node.lineno,
        )

    def _extract_params(self, args: ast.arguments) -> list[str]:
        """Extract parameter list with type annotations."""
        params = []

        # Positional-only params (before /)
        for arg in args.posonlyargs:
            params.append(self._format_arg(arg))

        if args.posonlyargs:
            params.append("/")

        # Regular positional/keyword params
        defaults_start = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            param = self._format_arg(arg)
            # Add default value indicator
            if i >= defaults_start:
                default_idx = i - defaults_start
                default = args.defaults[default_idx]
                param += f" = {self._node_to_str(default)}"
            params.append(param)

        # *args
        if args.vararg:
            params.append(f"*{self._format_arg(args.vararg)}")
        elif args.kwonlyargs:
            params.append("*")

        # Keyword-only params
        kw_defaults_map = {i: d for i, d in enumerate(args.kw_defaults) if d is not None}
        for i, arg in enumerate(args.kwonlyargs):
            param = self._format_arg(arg)
            if i in kw_defaults_map:
                param += f" = {self._node_to_str(kw_defaults_map[i])}"
            params.append(param)

        # **kwargs
        if args.kwarg:
            params.append(f"**{self._format_arg(args.kwarg)}")

        return params

    def _format_arg(self, arg: ast.arg) -> str:
        """Format a single argument with optional type annotation."""
        if arg.annotation:
            return f"{arg.arg}: {self._node_to_str(arg.annotation)}"
        return arg.arg

    def _node_to_str(self, node: ast.AST | None) -> str:
        """Convert AST node to string representation."""
        if node is None:
            return ""

        # Python 3.9+ has ast.unparse
        try:
            return ast.unparse(node)
        except AttributeError:
            # Fallback for older Python
            return self._manual_unparse(node)

    def _manual_unparse(self, node: ast.AST) -> str:
        """Manual unparse for older Python versions."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._manual_unparse(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._manual_unparse(node.value)}[{self._manual_unparse(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Tuple):
            elts = ", ".join(self._manual_unparse(e) for e in node.elts)
            return f"({elts})"
        elif isinstance(node, ast.List):
            elts = ", ".join(self._manual_unparse(e) for e in node.elts)
            return f"[{elts}]"
        elif isinstance(node, ast.BinOp):
            # Handle Union types like X | Y
            if isinstance(node.op, ast.BitOr):
                return f"{self._manual_unparse(node.left)} | {self._manual_unparse(node.right)}"
        elif isinstance(node, ast.Call):
            func = self._manual_unparse(node.func)
            args = ", ".join(self._manual_unparse(a) for a in node.args)
            return f"{func}({args})"

        return "<unknown>"


def extract_python(file_path: str | Path) -> ModuleInfo:
    """Convenience function to extract Python module info."""
    extractor = PythonASTExtractor()
    return extractor.extract(file_path)


def extract_file(file_path: str | Path) -> ModuleInfo:
    """
    Extract code structure from any supported file.

    Supports:
    - Python (.py, .pyx, .pyi) via native AST
    - TypeScript/JavaScript (.ts, .tsx, .js, .jsx) via tree-sitter
    - Go (.go) via tree-sitter-go
    - Rust (.rs) via tree-sitter-rust
    - Other languages via Pygments fallback (signatures only)
    """
    # Use HybridExtractor which handles all languages
    from tldr.hybrid_extractor import HybridExtractor

    extractor = HybridExtractor()
    return extractor.extract(file_path)


# === CLI ===

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ast_extractor.py <file_path> [--compact]")
        sys.exit(1)

    file_path = sys.argv[1]
    compact = "--compact" in sys.argv

    try:
        info = extract_file(file_path)
        output = info.to_compact() if compact else info.to_dict()
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
