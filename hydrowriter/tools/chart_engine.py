"""Chart generation utilities for HydroWriter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional dependency
    plt = None
    _MATPLOTLIB_ERROR = str(exc)
else:
    _MATPLOTLIB_ERROR = None
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


@dataclass(slots=True)
class ChartSeries:
    """Normalized series data for plotting."""

    x: list[Any]
    y: list[float]
    label: str = ""


class ChartEngine:
    """Generate charts with a unified blue hydro style."""

    _PALETTE = {
        "primary": "#0F5E9C",
        "secondary": "#2C7FB8",
        "accent": "#73B3D8",
        "grid": "#D7E8F3",
        "background": "#F7FBFE",
        "text": "#16324F",
        "success": "#4DA8DA",
    }

    _HYDRO_TYPES = {
        "water_level": ("line", "水位过程线", "时间", "水位 (m)"),
        "water_level_process": ("line", "水位过程线", "时间", "水位 (m)"),
        "flow": ("line", "流量过程线", "时间", "流量 (m^3/s)"),
        "flow_process": ("line", "流量过程线", "时间", "流量 (m^3/s)"),
        "discharge": ("line", "流量过程线", "时间", "流量 (m^3/s)"),
        "control_effect": ("bar", "控制效果图", "项目", "效果值"),
    }

    def __init__(self) -> None:
        self.matplotlib_available = plt is not None
        self.dependency_error = _MATPLOTLIB_ERROR

    def generate_line_chart(
        self,
        data: Any,
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: str,
    ) -> dict[str, Any]:
        """Generate a line chart."""
        series_list = self._normalize_series_list(data)
        if isinstance(series_list, dict):
            return series_list

        def renderer() -> tuple[Any, Any]:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for index, series in enumerate(series_list):
                color = self._series_color(index)
                ax.plot(
                    series.x,
                    series.y,
                    color=color,
                    linewidth=2.2,
                    marker="o",
                    markersize=4.5,
                    label=series.label or None,
                )
            self._style_axes(ax, title, xlabel, ylabel)
            if any(series.label for series in series_list):
                ax.legend(frameon=False)
            return fig, ax

        return self._render_chart("line", output_path, series_list, renderer)

    def generate_bar_chart(
        self,
        data: Any,
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: str,
    ) -> dict[str, Any]:
        """Generate a bar chart."""
        series_list = self._normalize_series_list(data)
        if isinstance(series_list, dict):
            return series_list

        def renderer() -> tuple[Any, Any]:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            categories = [str(item) for item in series_list[0].x]
            positions = list(range(len(categories)))
            group_count = len(series_list)
            width = 0.75 / max(group_count, 1)
            for index, series in enumerate(series_list):
                offset = (index - (group_count - 1) / 2) * width
                shifted = [value + offset for value in positions]
                ax.bar(
                    shifted,
                    series.y,
                    width=width,
                    label=series.label or None,
                    color=self._series_color(index),
                    edgecolor="white",
                    linewidth=0.8,
                )
            ax.set_xticks(positions)
            ax.set_xticklabels(categories)
            self._style_axes(ax, title, xlabel, ylabel)
            if any(series.label for series in series_list):
                ax.legend(frameon=False)
            return fig, ax

        return self._render_chart("bar", output_path, series_list, renderer)

    def generate_scatter_chart(
        self,
        data: Any,
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: str,
    ) -> dict[str, Any]:
        """Generate a scatter chart."""
        series_list = self._normalize_series_list(data)
        if isinstance(series_list, dict):
            return series_list

        def renderer() -> tuple[Any, Any]:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for index, series in enumerate(series_list):
                ax.scatter(
                    series.x,
                    series.y,
                    s=48,
                    alpha=0.85,
                    color=self._series_color(index),
                    edgecolors="white",
                    linewidths=0.8,
                    label=series.label or None,
                )
            self._style_axes(ax, title, xlabel, ylabel)
            if any(series.label for series in series_list):
                ax.legend(frameon=False)
            return fig, ax

        return self._render_chart("scatter", output_path, series_list, renderer)

    def generate_hydro_chart(
        self,
        data: Any,
        chart_type: str,
        output_path: str,
    ) -> dict[str, Any]:
        """Generate a hydro-specific chart."""
        chart_key = chart_type.strip().lower()
        if chart_key not in self._HYDRO_TYPES:
            return {
                "success": False,
                "chart_type": chart_type,
                "error": f"Unsupported hydro chart type: {chart_type}",
            }

        base_type, title, xlabel, ylabel = self._HYDRO_TYPES[chart_key]
        if base_type == "line":
            return self.generate_line_chart(data, title, xlabel, ylabel, output_path)
        return self.generate_bar_chart(data, title, xlabel, ylabel, output_path)

    def _render_chart(
        self,
        chart_type: str,
        output_path: str,
        series_list: list[ChartSeries],
        renderer: Any,
    ) -> dict[str, Any]:
        if not self.matplotlib_available or plt is None:
            return {
                "success": False,
                "chart_type": chart_type,
                "error": "matplotlib is not available.",
                "dependency_error": self.dependency_error,
            }
        if not output_path.strip():
            return {
                "success": False,
                "chart_type": chart_type,
                "error": "output_path is required.",
            }

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        fig = None
        try:
            fig, _ = renderer()
            fig.tight_layout()
            fig.savefig(destination, dpi=160, bbox_inches="tight")
            return {
                "success": True,
                "chart_type": chart_type,
                "output_path": str(destination),
                "series_count": len(series_list),
                "points": sum(len(series.y) for series in series_list),
            }
        except Exception as exc:
            return {
                "success": False,
                "chart_type": chart_type,
                "output_path": str(destination),
                "error": str(exc),
            }
        finally:
            if fig is not None:
                plt.close(fig)

    def _style_axes(self, ax: Any, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_title(title, fontsize=14, fontweight="bold", color=self._PALETTE["text"], pad=12)
        ax.set_xlabel(xlabel, color=self._PALETTE["text"])
        ax.set_ylabel(ylabel, color=self._PALETTE["text"])
        ax.set_facecolor(self._PALETTE["background"])
        ax.grid(True, color=self._PALETTE["grid"], linestyle="--", linewidth=0.8, alpha=0.9)
        ax.tick_params(colors=self._PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_color(self._PALETTE["accent"])
            spine.set_linewidth(0.9)

    def _series_color(self, index: int) -> str:
        colors = [
            self._PALETTE["primary"],
            self._PALETTE["secondary"],
            self._PALETTE["accent"],
            self._PALETTE["success"],
        ]
        return colors[index % len(colors)]

    def _normalize_series_list(self, data: Any) -> list[ChartSeries] | dict[str, Any]:
        if data is None:
            return {"success": False, "error": "Chart data is required."}

        if isinstance(data, Mapping):
            if "series" in data:
                raw_series = data["series"]
                if not isinstance(raw_series, Sequence) or isinstance(raw_series, (str, bytes)):
                    return {"success": False, "error": "data['series'] must be a sequence."}
                result: list[ChartSeries] = []
                for item in raw_series:
                    normalized = self._normalize_single_series(item)
                    if isinstance(normalized, dict):
                        return normalized
                    result.append(normalized)
                return self._validate_series_alignment(result)
            normalized = self._normalize_single_series(data)
            if isinstance(normalized, dict):
                return normalized
            return [normalized]

        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            normalized = self._normalize_single_series(data)
            if isinstance(normalized, dict):
                return normalized
            return [normalized]

        return {"success": False, "error": f"Unsupported chart data type: {type(data)!r}"}

    def _normalize_single_series(self, data: Any) -> ChartSeries | dict[str, Any]:
        if isinstance(data, Mapping):
            label = str(data.get("label", ""))
            if "x" in data and "y" in data:
                x_values = list(data["x"])
                y_values = list(data["y"])
            else:
                x_values = list(data.keys())
                y_values = list(data.values())
            return self._build_series(x_values, y_values, label)

        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            items = list(data)
            if not items:
                return {"success": False, "error": "Chart data sequence is empty."}

            first = items[0]
            if isinstance(first, Mapping) and "x" in first and "y" in first:
                x_values = [item["x"] for item in items]
                y_values = [item["y"] for item in items]
                return self._build_series(x_values, y_values, "")

            if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
                try:
                    x_values = [item[0] for item in items]
                    y_values = [item[1] for item in items]
                except Exception:
                    return {"success": False, "error": "Tuple data must contain x/y pairs."}
                return self._build_series(x_values, y_values, "")

            x_values = list(range(1, len(items) + 1))
            return self._build_series(x_values, items, "")

        return {"success": False, "error": f"Unsupported series data type: {type(data)!r}"}

    def _build_series(
        self,
        x_values: list[Any],
        y_values: list[Any],
        label: str,
    ) -> ChartSeries | dict[str, Any]:
        if len(x_values) != len(y_values):
            return {"success": False, "error": "x and y lengths must match."}
        if not x_values:
            return {"success": False, "error": "Chart data points are empty."}

        try:
            y_numbers = [float(value) for value in y_values]
        except (TypeError, ValueError) as exc:
            return {"success": False, "error": f"y values must be numeric: {exc}"}

        return ChartSeries(x=list(x_values), y=y_numbers, label=label)

    def _validate_series_alignment(
        self, series_list: list[ChartSeries]
    ) -> list[ChartSeries] | dict[str, Any]:
        if not series_list:
            return {"success": False, "error": "Series data is empty."}
        baseline = series_list[0].x
        for series in series_list[1:]:
            if len(series.x) != len(baseline):
                return {"success": False, "error": "All series must have the same length."}
        return series_list
