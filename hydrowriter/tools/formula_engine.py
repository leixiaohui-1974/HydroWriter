"""Formula verification and formatting utilities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

try:
    import sympy as sp
    from sympy.core.relational import Relational

    try:
        from sympy.parsing.latex import parse_latex as sympy_parse_latex
        _LATEX_PARSER_ERROR: str | None = None
    except Exception as exc:  # pragma: no cover - depends on optional antlr runtime
        sympy_parse_latex = None
        _LATEX_PARSER_ERROR = str(exc)
except Exception as exc:  # pragma: no cover - optional dependency
    sp = None
    Relational = Any  # type: ignore[assignment]
    sympy_parse_latex = None
    _LATEX_PARSER_ERROR = str(exc)


DimensionMap = dict[str, Any]


class FormulaEngine:
    """Validate LaTeX formulas and perform basic dimension checks."""

    _LATEX_COMMANDS = {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "theta",
        "lambda",
        "mu",
        "nu",
        "omega",
        "phi",
        "pi",
        "rho",
        "sigma",
        "tau",
        "chi",
        "psi",
        "eta",
        "kappa",
        "zeta",
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "log",
        "ln",
        "exp",
        "frac",
        "sqrt",
        "left",
        "right",
        "cdot",
        "times",
        "text",
        "mathrm",
        "operatorname",
        "begin",
        "end",
        "displaystyle",
    }

    def __init__(self) -> None:
        self.sympy_available = sp is not None
        self.latex_parser_available = sympy_parse_latex is not None
        self.dependency_error = _LATEX_PARSER_ERROR

    def verify_latex(self, formula: str) -> dict[str, Any]:
        """Verify whether a LaTeX formula is parseable and mathematically well formed."""
        cleaned = self._clean_formula(formula)
        if not cleaned:
            return {
                "success": False,
                "valid": False,
                "degraded": False,
                "error": "Formula is empty.",
                "symbols": [],
            }

        if not self.sympy_available:
            return {
                "success": False,
                "valid": False,
                "degraded": True,
                "error": "sympy is not available.",
                "symbols": self.extract_symbols(cleaned),
            }

        try:
            expression = self._parse_formula(cleaned)
            is_equation = isinstance(expression, Relational)
            result: dict[str, Any] = {
                "success": True,
                "valid": True,
                "degraded": False,
                "is_equation": is_equation,
                "symbols": sorted(str(symbol) for symbol in expression.free_symbols),
                "normalized": self._to_latex(expression),
                "expression": str(expression),
            }
            if is_equation:
                result["symbolic_identity"] = self._is_symbolic_identity(expression)
            else:
                result["simplified"] = str(sp.simplify(expression))
            return result
        except Exception as exc:
            error_text = str(exc)
            is_dependency_issue = (
                "antlr4" in error_text.lower()
                or "latex parser" in error_text.lower()
                or not self.latex_parser_available
            )
            return {
                "success": False,
                "valid": False,
                "degraded": is_dependency_issue,
                "error": error_text,
                "symbols": self.extract_symbols(cleaned),
            }

    def extract_symbols(self, formula: str) -> list[str]:
        """Extract symbols from a LaTeX formula."""
        cleaned = self._clean_formula(formula)
        if not cleaned:
            return []

        if self.sympy_available:
            try:
                expression = self._parse_formula(cleaned)
                return sorted(str(symbol) for symbol in expression.free_symbols)
            except Exception:
                pass

        tokens = re.findall(r"\\[A-Za-z]+|[A-Za-z]+(?:_[A-Za-z0-9]+)?", cleaned)
        symbols = {
            token.lstrip("\\")
            for token in tokens
            if token.lstrip("\\") not in self._LATEX_COMMANDS
        }
        return sorted(symbols)

    def check_dimensions(self, formula: str, dimensions: dict[str, Any]) -> bool:
        """Perform dimension analysis on a formula.

        The ``dimensions`` mapping accepts values like ``"L^3/T"`` or
        ``{"L": 3, "T": -1}``.
        """
        cleaned = self._clean_formula(formula)
        if not cleaned or not dimensions or not self.sympy_available:
            return False

        try:
            expression = self._parse_formula(cleaned)
            context = {
                name: self._coerce_dimension(value)
                for name, value in dimensions.items()
            }
            if isinstance(expression, Relational):
                left = self._dimension_of(expression.lhs, context)
                right = self._dimension_of(expression.rhs, context)
                return self._same_dimension(left, right)
            self._dimension_of(expression, context)
            return True
        except Exception:
            return False

    def format_equation(self, formula: str, style: str = "display") -> str:
        """Format a formula for output."""
        cleaned = self._clean_formula(formula)
        if not cleaned:
            return ""

        normalized = cleaned
        if self.sympy_available:
            try:
                normalized = self._to_latex(self._parse_formula(cleaned))
            except Exception:
                normalized = cleaned

        style_key = style.strip().lower()
        if style_key == "inline":
            return f"${normalized}$"
        if style_key == "equation":
            return f"\\begin{{equation}}\n{normalized}\n\\end{{equation}}"
        if style_key == "plain":
            return normalized
        return f"$$\n{normalized}\n$$"

    def _parse_formula(self, formula: str) -> Any:
        if not self.sympy_available:
            raise RuntimeError("sympy is not available.")
        if not self.latex_parser_available or sympy_parse_latex is None:
            raise RuntimeError(
                "sympy LaTeX parser is not available."
                + (f" {_LATEX_PARSER_ERROR}" if _LATEX_PARSER_ERROR else "")
            )
        return sympy_parse_latex(formula)

    def _clean_formula(self, formula: str) -> str:
        cleaned = formula.strip()
        if cleaned.startswith("$$") and cleaned.endswith("$$"):
            cleaned = cleaned[2:-2].strip()
        elif cleaned.startswith("$") and cleaned.endswith("$"):
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def _to_latex(self, expression: Any) -> str:
        if isinstance(expression, Relational):
            return f"{sp.latex(expression.lhs)} = {sp.latex(expression.rhs)}"
        return sp.latex(expression)

    def _is_symbolic_identity(self, expression: Any) -> bool | None:
        if not isinstance(expression, Relational):
            return None
        try:
            difference = sp.simplify(expression.lhs - expression.rhs)
        except Exception:
            return None
        return difference == 0

    def _coerce_dimension(self, value: Any) -> DimensionMap:
        if isinstance(value, Mapping):
            return {
                str(key): sp.sympify(power)
                for key, power in value.items()
                if sp.sympify(power) != 0
            }
        if isinstance(value, (int, float)):
            return {}
        if isinstance(value, str):
            text = value.strip()
            if not text or text == "1":
                return {}
            return self._parse_dimension_expression(text)
        raise TypeError(f"Unsupported dimension format: {type(value)!r}")

    def _parse_dimension_expression(self, expression: str) -> DimensionMap:
        base_tokens = {
            token: {token: sp.Integer(1)}
            for token in set(re.findall(r"[A-Za-z]+", expression))
        }
        prepared = expression.replace("^", "**")
        prepared = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z(])", "*", prepared)
        parsed = sp.sympify(prepared, locals={name: sp.Symbol(name) for name in base_tokens})
        return self._dimension_of(parsed, base_tokens)

    def _dimension_of(self, expression: Any, context: dict[str, DimensionMap]) -> DimensionMap:
        if expression.is_Number:
            return {}
        if expression.is_Symbol:
            name = str(expression)
            if name not in context:
                raise KeyError(f"Missing dimension for symbol: {name}")
            return dict(context[name])
        if expression.is_Add:
            dims = [self._dimension_of(arg, context) for arg in expression.args]
            first = dims[0]
            for other in dims[1:]:
                if not self._same_dimension(first, other):
                    raise ValueError("Addition/subtraction requires matching dimensions.")
            return first
        if expression.is_Mul:
            combined: DimensionMap = {}
            for arg in expression.args:
                for key, power in self._dimension_of(arg, context).items():
                    combined[key] = sp.simplify(combined.get(key, 0) + power)
            return {key: value for key, value in combined.items() if value != 0}
        if expression.is_Pow:
            exponent = sp.sympify(expression.exp)
            if not exponent.is_number:
                raise ValueError("Dimension exponents must be numeric.")
            return {
                key: sp.simplify(power * exponent)
                for key, power in self._dimension_of(expression.base, context).items()
                if sp.simplify(power * exponent) != 0
            }
        if expression.func in {sp.Abs}:
            return self._dimension_of(expression.args[0], context)
        if expression.func in {sp.Min, sp.Max}:
            dims = [self._dimension_of(arg, context) for arg in expression.args]
            first = dims[0]
            for other in dims[1:]:
                if not self._same_dimension(first, other):
                    raise ValueError("Min/Max requires matching dimensions.")
            return first
        if getattr(expression, "is_Function", False):
            for arg in expression.args:
                if not self._same_dimension(self._dimension_of(arg, context), {}):
                    raise ValueError("Function arguments must be dimensionless.")
            return {}
        raise ValueError(f"Unsupported expression for dimension analysis: {expression!r}")

    def _same_dimension(self, left: DimensionMap, right: DimensionMap) -> bool:
        keys = set(left) | set(right)
        return all(sp.simplify(left.get(key, 0) - right.get(key, 0)) == 0 for key in keys)
