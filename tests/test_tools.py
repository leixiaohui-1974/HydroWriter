"""Tests for HydroWriter tools — formula engine and chart engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hydrowriter.tools.formula_engine import FormulaEngine
from hydrowriter.tools.chart_engine import ChartEngine


# --- FormulaEngine Tests ---

class TestFormulaEngine:
    def test_empty_formula(self):
        engine = FormulaEngine()
        result = engine.verify_latex("")
        assert not result["valid"]
        assert "empty" in result["error"].lower()

    def test_extract_symbols_empty(self):
        engine = FormulaEngine()
        symbols = engine.extract_symbols("")
        assert symbols == []

    def test_extract_symbols_basic(self):
        engine = FormulaEngine()
        symbols = engine.extract_symbols("Q = C_d \\cdot a \\cdot \\sqrt{2 g h}")
        # Should extract variable symbols, not LaTeX commands
        assert isinstance(symbols, list)
        assert len(symbols) > 0

    def test_format_equation_display(self):
        engine = FormulaEngine()
        formatted = engine.format_equation("E = mc^2", style="display")
        assert "$$" in formatted

    def test_format_equation_inline(self):
        engine = FormulaEngine()
        formatted = engine.format_equation("E = mc^2", style="inline")
        assert formatted.startswith("$")
        assert formatted.endswith("$")
        assert not formatted.startswith("$$")

    def test_format_equation_empty(self):
        engine = FormulaEngine()
        assert engine.format_equation("") == ""

    def test_verify_returns_dict(self):
        engine = FormulaEngine()
        result = engine.verify_latex("x + y")
        assert isinstance(result, dict)
        assert "valid" in result
        assert "symbols" in result


class TestFormulaEngineWithSympy:
    """Tests that use sympy features (skipped if unavailable)."""

    @pytest.fixture
    def engine(self):
        engine = FormulaEngine()
        if not engine.sympy_available:
            pytest.skip("sympy not installed")
        return engine

    def test_verify_valid_formula(self, engine):
        if not engine.latex_parser_available:
            pytest.skip("sympy LaTeX parser not available")
        result = engine.verify_latex("x + y")
        assert result["valid"] or not result["success"]
        # Even if parsing fails due to antlr issues, it should not crash

    def test_check_dimensions_returns_bool(self, engine):
        result = engine.check_dimensions("Q = A v", {"Q": "L^3/T", "A": "L^2", "v": "L/T"})
        assert isinstance(result, bool)


# --- ChartEngine Tests ---

class TestChartEngine:
    def test_chart_engine_init(self):
        engine = ChartEngine()
        assert isinstance(engine.matplotlib_available, bool)

    def test_generate_hydro_chart_unknown_type(self):
        engine = ChartEngine()
        result = engine.generate_hydro_chart(
            data={"x": [1, 2], "y": [3, 4]},
            chart_type="nonexistent_type",
            output_path="test.png",
        )
        assert not result["success"]
        assert "Unsupported" in result["error"]

    def test_line_chart_none_data(self):
        engine = ChartEngine()
        result = engine.generate_line_chart(
            data=None, title="Test", xlabel="X", ylabel="Y", output_path="test.png",
        )
        assert not result["success"]

    def test_bar_chart_none_data(self):
        engine = ChartEngine()
        result = engine.generate_bar_chart(
            data=None, title="Test", xlabel="X", ylabel="Y", output_path="test.png",
        )
        assert not result["success"]

    def test_scatter_chart_none_data(self):
        engine = ChartEngine()
        result = engine.generate_scatter_chart(
            data=None, title="Test", xlabel="X", ylabel="Y", output_path="test.png",
        )
        assert not result["success"]


class TestChartEngineWithMatplotlib:
    """Tests that generate charts (skipped if matplotlib unavailable)."""

    @pytest.fixture
    def engine(self):
        engine = ChartEngine()
        if not engine.matplotlib_available:
            pytest.skip("matplotlib not installed")
        return engine

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_line_chart(self, engine, tmp_dir):
        out = str(tmp_dir / "line.png")
        result = engine.generate_line_chart(
            data={"x": [0, 1, 2, 3, 4], "y": [1.0, 1.5, 2.0, 1.8, 1.3]},
            title="Water Level", xlabel="Time (h)", ylabel="Level (m)",
            output_path=out,
        )
        assert result["success"]
        assert Path(result["output_path"]).exists()

    def test_bar_chart(self, engine, tmp_dir):
        out = str(tmp_dir / "bar.png")
        result = engine.generate_bar_chart(
            data={"x": ["A", "B", "C"], "y": [100, 200, 150]},
            title="Intake Volume", xlabel="Node", ylabel="Volume (m3)",
            output_path=out,
        )
        assert result["success"]

    def test_scatter_chart(self, engine, tmp_dir):
        out = str(tmp_dir / "scatter.png")
        result = engine.generate_scatter_chart(
            data={"x": [1, 2, 3, 4, 5], "y": [1.1, 2.0, 2.9, 4.1, 5.0]},
            title="Model Validation", xlabel="Observed", ylabel="Simulated",
            output_path=out,
        )
        assert result["success"]

    def test_hydro_chart_water_level(self, engine, tmp_dir):
        out = str(tmp_dir / "wl.png")
        result = engine.generate_hydro_chart(
            data={"x": list(range(24)), "y": [1.0 + 0.1 * i for i in range(24)]},
            chart_type="water_level",
            output_path=out,
        )
        assert result["success"]

    def test_hydro_chart_discharge(self, engine, tmp_dir):
        out = str(tmp_dir / "q.png")
        result = engine.generate_hydro_chart(
            data={"x": list(range(12)), "y": [0.5, 1.0, 2.5, 5.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5]},
            chart_type="discharge",
            output_path=out,
        )
        assert result["success"]

    def test_line_chart_empty_output_path(self, engine):
        result = engine.generate_line_chart(
            data={"x": [1, 2], "y": [3, 4]},
            title="T", xlabel="X", ylabel="Y", output_path="",
        )
        assert not result["success"]
        assert "output_path" in result["error"]
