"""Utility functions to manage subplots."""

from __future__ import annotations

import functools
import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Generator
from enum import Enum
from typing import Any

import numpy as np
import plotly
import plotly.graph_objs as go
from numpy.typing import NDArray
from plotly.exceptions import PlotlyKeyError
from pydantic import ValidationInfo, field_validator

from statsplotly import constants
from statsplotly._base import BaseModel
from statsplotly.exceptions import StatsPlotSpecificationError
from statsplotly.plot_objects.trace import BaseTrace
from statsplotly.plot_specifiers.common import smart_legend
from statsplotly.plot_specifiers.data import DataDimension
from statsplotly.plot_specifiers.layout import (
    AxesSpecifier,
    ColoraxisReference,
    PlotAxis,
)

logger = logging.getLogger(__name__)


class GridAxis(str, Enum):
    COLS = "cols"
    ROWS = "rows"


class SharedGridAxis(str, Enum):
    COLS = "cols"
    ROWS = "rows"
    ALL = "all"


AXIS_TO_DATA_MAP = {PlotAxis.XAXIS: DataDimension.X, PlotAxis.YAXIS: DataDimension.Y}


class FigureLayoutFormatter(BaseModel):
    fig: go.Figure

    @property
    def coloraxis_references(self) -> NDArray[Any]:
        return np.sort(
            [
                int(ref) if ref else 1
                for ref in [
                    key.split("coloraxis")[-1]
                    for key in self.fig.layout
                    if key.startswith("coloraxis")
                ]
            ]
        )

    def hide_legend_duplicates(self) -> None:
        names = set()
        self.fig.for_each_trace(
            lambda trace: (
                trace.update(showlegend=False) if trace.name in names else names.add(trace.name)
            )
        )

    def update_coloraxis_reference(
        self,
        coloraxis_reference: ColoraxisReference,
        layout_dict: dict[str, Any],
        traces: dict[str, BaseTrace],
    ) -> None:
        coloraxis_refs = self.coloraxis_references[::-1]
        if len(coloraxis_refs) > 0:
            new_coloraxis_ref = "".join(
                (
                    coloraxis_reference,
                    str(coloraxis_refs[0] + 1),
                )
            )
            layout_dict[new_coloraxis_ref] = layout_dict.pop(coloraxis_reference)
            # Update traces objects
            for trace in traces.values():
                if hasattr(trace, "coloraxis"):
                    if trace.coloraxis is not None:
                        trace.coloraxis = new_coloraxis_ref

                if hasattr(trace, "marker"):
                    if hasattr(trace.marker, "coloraxis"):
                        if trace.marker["coloraxis"] is not None:
                            trace.marker = trace.marker.update({"coloraxis": new_coloraxis_ref})


class _SubplotGridValidator(FigureLayoutFormatter):

    @field_validator("fig")
    def validate_subplot_grid(cls, fig: go.Figure) -> go.Figure:
        if fig._grid_ref is None:
            raise StatsPlotSpecificationError(
                "Use `plotly.tools.make_subplots` to create a figure with a subplot grid"
            )
        return fig


class FigureSubplotFormatter(_SubplotGridValidator):
    row: int
    col: int

    @field_validator("row", "col", mode="before")
    def validate_grid_indices(cls, value: int | None) -> int:
        if value is None:
            return 1
        return value

    def get_n_rows_from_col_number(self) -> int:
        return len([row for row in self.fig._grid_ref if row[self.col - 1] is not None])

    def get_n_cols_from_row_number(self) -> int:
        return len([subplot for subplot in self.fig._grid_ref[self.row - 1] if subplot is not None])

    def get_axis_domain(self, plot_axis: PlotAxis) -> list[float]:
        if plot_axis is PlotAxis.XAXIS:
            dimension_idx = 0
        elif plot_axis is PlotAxis.YAXIS:
            dimension_idx = 1
        else:
            raise NotImplementedError(f"Method not implemented for `plot_axis={plot_axis.value}`")

        return self.fig.layout[
            self.fig._grid_ref[self.row - 1][self.col - 1][0].layout_keys[dimension_idx]
        ]["domain"]

    def format_colorbar_in_subplots_layout(
        self,
        coloraxis_dict: dict[str, Any],
    ) -> None:
        if self.get_n_rows_from_col_number() > 1:
            if coloraxis_dict["colorbar"] is not None:
                _axis_domain = self.get_axis_domain(plot_axis=PlotAxis.YAXIS)
                coloraxis_dict["colorbar"]["y"] = _axis_domain[0]
                coloraxis_dict["colorbar"]["yanchor"] = "bottom"
                coloraxis_dict["colorbar"]["len"] = _axis_domain[1] - _axis_domain[0]

        if self.get_n_cols_from_row_number() > 1:
            if coloraxis_dict["colorbar"] is not None:
                _axis_domain = self.get_axis_domain(plot_axis=PlotAxis.XAXIS)
                coloraxis_dict["colorbar"]["x"] = _axis_domain[1]
                coloraxis_dict["colorbar"]["thickness"] = np.max(
                    [
                        (_axis_domain[1] - _axis_domain[0])
                        * constants.COLORBAR_THICKNESS_SCALING_FACTOR,
                        5,
                    ]
                )


class _SubplotGridCommonAxisFormatter(BaseModel, metaclass=ABCMeta):
    fig: go.Figure
    plot_axis: PlotAxis | None
    shared_grid_axis: SharedGridAxis

    @property
    def dimension(self) -> PlotAxis | None:
        if self.plot_axis is None:
            if self.shared_grid_axis is SharedGridAxis.COLS:
                return PlotAxis.XAXIS
            if self.shared_grid_axis is SharedGridAxis.ROWS:
                return PlotAxis.YAXIS

        return self.plot_axis

    @property
    def _plot_groups(self) -> list[list[plotly.subplots.SubplotRef]]:
        if self.shared_grid_axis == GridAxis.COLS:  # type: ignore[comparison-overlap]
            return [
                [row[i] for row in self.fig._grid_ref] for i in range(len(self.fig._grid_ref[0]))
            ]

        if self.shared_grid_axis == GridAxis.ROWS:  # type: ignore[comparison-overlap]
            return self.fig._grid_ref

        return [[fig_ref for ref_group in self.fig._grid_ref for fig_ref in ref_group]]

    @property
    def col_dimension_size(self) -> int | None:
        if self.shared_grid_axis is SharedGridAxis.ALL:
            return None
        return len(self.fig._grid_ref[0])

    @property
    def row_dimension_size(self) -> int | None:
        if self.shared_grid_axis is SharedGridAxis.ALL:
            return None
        return len(self.fig._grid_ref[0][0])

    def iter_plot_groups(self) -> Generator[list[tuple[plotly.subplots.SubplotRef]], None, None]:
        for plot_group in self._plot_groups:
            yield [subplot for subplot in plot_group if subplot is not None]

    def get_plot_group_axes_reference(
        self, plot_group: list[plotly.subplots.SubplotRef]
    ) -> list[dict[str, Any]]:
        return [axes[0].trace_kwargs for axes in plot_group]

    def get_target_axes_from_target_traces(
        self, target_traces: list[plotly.basedatatypes.BaseTraceType]
    ) -> list[str]:
        if self.dimension is PlotAxis.XAXIS:
            idx = 0
        elif self.dimension is PlotAxis.YAXIS:
            idx = 1
        else:
            raise StatsPlotSpecificationError(
                f"Can not get target axes for plot_axis={self.plot_axis}"
            )

        return [
            axes[0].layout_keys[idx]
            for plot_group in self.iter_plot_groups()
            for axes in plot_group
            if axes[0].trace_kwargs[self.dimension.value]
            in [trace[self.dimension.value] for trace in target_traces]
        ]

    @abstractmethod
    def get_target_traces(
        self, plot_group: list[plotly.subplots.SubplotRef]
    ) -> list[plotly.basedatatypes.BaseTraceType]:
        """This method returns a list of plotly traces belonging to the `plot_group` and relevant to the formatter."""

    @abstractmethod
    def update_traces_and_layout(
        self, target_traces: list[plotly.basedatatypes.BaseTraceType]
    ) -> None:
        """This method updates traces and layout attributes given the formatting arguments."""

    def update_along_grid_axis(self) -> None:
        for plot_group in self.iter_plot_groups():
            target_traces = self.get_target_traces(plot_group)
            if len(target_traces) == 0:
                continue

            self.update_traces_and_layout(target_traces)


class _SubplotGridCommonColoraxisFormatter(_SubplotGridCommonAxisFormatter):
    plot_axis: PlotAxis = PlotAxis.COLORAXIS

    @staticmethod
    def _get_heatmap_trace_colorlimit(
        trace: plotly.basedatatypes.BaseTraceType,
    ) -> tuple[float, float]:
        if trace.zmin is not None:
            z_min = trace.zmin
        else:
            z_min = np.min(trace.z)

        if trace.zmax is not None:
            z_max = trace.zmax
        else:
            z_max = np.max(trace.z)

        return z_min, z_max

    @staticmethod
    def _get_scatter_trace_colorlimit(
        trace: plotly.basedatatypes.BaseTraceType,
    ) -> tuple[float, float]:

        return np.min(trace.marker.color), np.max(trace.marker.color)

    def get_target_traces(
        self, plot_group: list[plotly.subplots.SubplotRef]
    ) -> list[plotly.basedatatypes.BaseTraceType]:
        axes_reference = self.get_plot_group_axes_reference(plot_group=plot_group)

        return [
            trace
            for trace in self.fig.data
            if any(
                [  # noqa: C419
                    trace[key] == value
                    for trace_axis_reference in axes_reference
                    for key, value in trace_axis_reference.items()
                ]
            )
            and (PlotAxis.COLORAXIS in trace or trace.marker[PlotAxis.COLORAXIS] is not None)
        ]

    def update_traces_and_layout(
        self, target_traces: list[plotly.basedatatypes.BaseTraceType]
    ) -> None:
        # Update traces
        try:
            reference_coloraxis = target_traces[-1][PlotAxis.COLORAXIS]
            color_limit_function = self._get_heatmap_trace_colorlimit
            self.fig.for_each_trace(
                lambda trace, reference_coloraxis=reference_coloraxis, target_traces=target_traces: (
                    trace.update(coloraxis=reference_coloraxis) if trace in target_traces else ()
                )
            )
        except PlotlyKeyError:
            reference_coloraxis = target_traces[-1].marker[PlotAxis.COLORAXIS]
            color_limit_function = self._get_scatter_trace_colorlimit
            self.fig.for_each_trace(
                lambda trace, reference_coloraxis=reference_coloraxis, target_traces=target_traces: (
                    trace.update(marker_coloraxis=reference_coloraxis)
                    if trace in target_traces
                    else ()
                )
            )

        # Update layout
        color_limits = [color_limit_function(trace) for trace in target_traces]
        self.fig.update_layout(
            {
                reference_coloraxis: {
                    "cmin": np.min([limit[0] for limit in color_limits]),
                    "cmax": np.max([limit[1] for limit in color_limits]),
                }
            }
        )

        # Update colorbar
        if "colorbar" in self.fig.layout[reference_coloraxis]:
            figure_subplot_formatter = FigureSubplotFormatter(
                fig=self.fig, row=self.row_dimension_size, col=self.col_dimension_size
            )
            if self.shared_grid_axis in (SharedGridAxis.COLS, SharedGridAxis.ALL):
                layout_update = {
                    reference_coloraxis: {
                        "colorbar": {
                            "orientation": "h",
                            "y": constants.COLORBAR_BOTTOM_POSITION,
                            "xanchor": (
                                "right" if ((self.col_dimension_size or 1) > 1) else "center"
                            ),
                            "len": np.abs(
                                np.subtract(
                                    *figure_subplot_formatter.get_axis_domain(
                                        plot_axis=PlotAxis.YAXIS
                                    )
                                )  # type: ignore
                            ),
                        },
                    }
                }
            elif self.shared_grid_axis is SharedGridAxis.ROWS:
                layout_update = {
                    reference_coloraxis: {
                        "colorbar": {"thickness": constants.COLORBAR_THICKNESS_SCALING_FACTOR},
                    }
                }
                if any(trace.showlegend or True for trace in self.fig.data):
                    layout_update[reference_coloraxis]["colorbar"].update({"yanchor": "top"})
            self.fig.update_layout(layout_update)


class _SubplotGridCommonXYAxisFormatter(_SubplotGridCommonAxisFormatter):
    common_range: bool
    link_axes: bool

    @field_validator("shared_grid_axis", mode="after")
    def check_shared_grid_axis(cls, value: SharedGridAxis, info: ValidationInfo) -> SharedGridAxis:
        if value is SharedGridAxis.ALL and info.data.get("plot_axis") is None:
            raise StatsPlotSpecificationError(
                f"`plot_axis` must be specified when using `shared_grid_axis = {SharedGridAxis.ALL.value}`"
            )
        return value

    @staticmethod
    def _check_numeric_cast(data: NDArray[Any]) -> bool:
        try:
            data.astype("float")
            return True
        except (ValueError, AttributeError):
            return False

    def _get_trace_axis_limit(
        self, trace: plotly.basedatatypes.BaseTraceType
    ) -> tuple[float, float] | None:

        if (trace_data := trace[AXIS_TO_DATA_MAP[self.dimension]]) is None:  # type: ignore
            logger.info(f"Axis limits of {trace.name} of type {type(trace)} can not be extracted")
            return None

        if not self._check_numeric_cast(trace_data):
            logger.debug(
                f"Can not be cast {trace.name} data of type {trace_data.dtype} to numeric dtype"
            )
            return None

        sanitized_trace_data = [datum for datum in trace_data if datum is not None]
        if len(sanitized_trace_data) == 0:
            return None

        return np.min(sanitized_trace_data), np.max(sanitized_trace_data)

    def get_target_traces(
        self, plot_group: list[plotly.subplots.SubplotRef]
    ) -> list[plotly.basedatatypes.BaseTraceType]:
        axes_reference = self.get_plot_group_axes_reference(plot_group=plot_group)
        if self.dimension not in [PlotAxis.XAXIS, PlotAxis.YAXIS]:
            raise StatsPlotSpecificationError(
                f"Unrecognized {self.dimension} can not be formatted across grid"
            )

        return [
            trace
            for trace in self.fig.data
            if trace[self.dimension.value]
            in [
                axis
                for trace_axis_reference in axes_reference
                for axis in trace_axis_reference.values()
            ]
            and not isinstance(trace, go.Bar)
        ]

    def _compute_axis_range(
        self,
        axis_limits: list[tuple[float, float] | None],
    ) -> list[float] | None:
        try:
            min_value, max_value = [
                np.min([limit[0] if limit is not None else None for limit in axis_limits]),
                np.max([limit[1] if limit is not None else None for limit in axis_limits]),
            ]

            return AxesSpecifier.pad_axis_range(
                axis_range=[min_value, max_value], padding_factor=constants.RANGE_PADDING_FACTOR
            )

        except (ValueError, TypeError):
            # No computable limits
            return None

    def update_traces_and_layout(
        self, target_traces: list[plotly.basedatatypes.BaseTraceType]
    ) -> None:
        # Update axis limits
        range_dict: dict[str, list[float] | None] = {"range": None}
        if self.common_range:
            axis_limits = [self._get_trace_axis_limit(trace) for trace in target_traces]
            axis_range = self._compute_axis_range(axis_limits)
            range_dict.update({"range": axis_range})

        self.fig.update_layout(
            {
                axis_reference: range_dict
                for axis_reference in self.get_target_axes_from_target_traces(target_traces)
            }
        )

        # Optionally link axes
        match_dict: dict[str, str | None] = {"matches": None}
        if self.link_axes:
            match_dict.update({"matches": target_traces[-1][self.dimension]})

        self.fig.update_layout(
            {
                axis_reference: match_dict
                for axis_reference in self.get_target_axes_from_target_traces(target_traces)
            }
        )


class SubplotGridFormatter(_SubplotGridValidator):

    def _get_row_yaxes_references(self, row_idx: int) -> list[str]:
        return [col[0].layout_keys[1] for col in self.fig._grid_ref[row_idx]]

    def _get_col_xaxes_references(self, col_idx: int) -> list[str]:
        return [row[col_idx][0].layout_keys[0] for row in self.fig._grid_ref]

    def _reduce_axis_ticks_union(
        self, axes_reference_function: Callable[[int], list[str]]
    ) -> Callable[[int], list[str]]:

        def _get_common_ticks(subplot_idx: int) -> list[str]:
            return list(
                functools.reduce(
                    lambda a, b: set(a) & set(b),
                    [
                        self.fig.layout[axis_ref]["ticktext"]
                        for axis_ref in axes_reference_function(subplot_idx)
                        if self.fig.layout[axis_ref]["ticktext"] is not None
                    ],
                )
            )

        return _get_common_ticks

    def check_common_xaxes_ticks(self, col_idx: int) -> bool:
        try:
            if len(self._reduce_axis_ticks_union(self._get_col_xaxes_references)(col_idx)) == 0:
                return False
            return True
        except TypeError:
            return True

    def check_common_yaxes_ticks(self, row_idx: int) -> bool:
        try:
            if len(self._reduce_axis_ticks_union(self._get_row_yaxes_references)(row_idx)) == 0:
                return False
            return True
        except TypeError:
            return True

    def set_common_coloraxis(self, shared_grid_axis: str) -> SubplotGridFormatter:
        _SubplotGridCommonColoraxisFormatter(
            fig=self.fig, shared_grid_axis=shared_grid_axis
        ).update_along_grid_axis()

        return self

    def set_common_axis_limit(
        self,
        shared_grid_axis: str = SharedGridAxis.ALL,
        plot_axis: str | None = None,
        common_range: bool = True,
        link_axes: bool = False,
    ) -> SubplotGridFormatter:
        _SubplotGridCommonXYAxisFormatter(
            fig=self.fig,
            shared_grid_axis=shared_grid_axis,
            plot_axis=plot_axis,
            common_range=common_range,
            link_axes=link_axes,
        ).update_along_grid_axis()

        return self

    def set_suplotgrid_titles(self, grid_axis: GridAxis) -> Callable[..., Any]:
        annotation_dict = {
            "font": constants.AXIS_TITLEFONT,
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
        }

        if grid_axis is GridAxis.COLS:
            grid_axis_length = len(self.fig._grid_ref[0])
            text_coordinates = "x"
            annotation_dict.update({"xanchor": "center", "y": 1, "yanchor": "top", "yshift": +30})
            axes_domain = [
                self.fig.layout[axes[0].layout_keys[0]]["domain"] for axes in self.fig._grid_ref[0]
            ]
        elif grid_axis is GridAxis.ROWS:
            # Add some left margin
            try:
                self.fig.update_layout({"margin_l": self.fig.layout.margin.l + 10})
            except TypeError:
                self.fig.layout.margin.l = 200

            grid_axis_length = len(self.fig._grid_ref)
            text_coordinates = "y"
            annotation_dict.update(
                {
                    "x": 0,
                    "xanchor": "right",
                    "xshift": -80,
                    "yanchor": "middle",
                }
            )
            axes_domain = [
                self.fig.layout[row[0][0].layout_keys[1]]["domain"] for row in self.fig._grid_ref
            ]

        def _apply_grid_titles(titles: str) -> None:
            if grid_axis_length != len(titles):
                raise StatsPlotSpecificationError(
                    f"Received {len(titles)} subplot titles for a {grid_axis.value} of length = {grid_axis_length}"
                )

            for i, title in enumerate(titles, 1):
                annotation_dict.update(
                    {"text": smart_legend(title), text_coordinates: np.mean(axes_domain[i - 1])}
                )
                self.fig.add_annotation(annotation_dict)

        return _apply_grid_titles

    def tidy_axes(self) -> None:
        for row, subplot_row in enumerate(self.fig._grid_ref):
            for col, subplot_col in enumerate(subplot_row):
                if subplot_col is None:
                    continue

                xaxis_ref, yaxis_ref = subplot_col[0].layout_keys
                if (
                    row < len(self.fig._grid_ref) - 1
                    and self.fig.layout[xaxis_ref].matches is not None
                ):

                    self.fig.update_layout(
                        {
                            xaxis_ref: {
                                "title": None,
                                "showticklabels": (
                                    False if self.check_common_xaxes_ticks(col_idx=col) else True
                                ),
                            }
                        }
                    )

                if col > 0 and self.fig.layout[yaxis_ref].matches is not None:

                    self.fig.update_layout(
                        {
                            yaxis_ref: {
                                "title": None,
                                "showticklabels": (
                                    False if self.check_common_yaxes_ticks(row_idx=row) else True
                                ),
                            }
                        }
                    )

    def tidy_subplots(
        self,
        title: str | None = None,
        no_legend: bool = False,
        row_titles: list[str] | None = None,
        col_titles: list[str] | None = None,
    ) -> SubplotGridFormatter:
        """Tidy a subplot grid by removing redundant axis titles and optionnaly adding annotations."""

        # Replace title if supplied
        if title is not None:
            self.fig.update_layout(title=title)

        # Clean legend
        if no_legend:
            self.fig.update_layout({"showlegend": False})
        else:
            # Remove legend duplicates
            self.hide_legend_duplicates()

        # Tidy axes
        self.tidy_axes()

        # Grid naming
        if col_titles is not None:
            self.set_suplotgrid_titles(grid_axis=GridAxis.COLS)(col_titles)

        if row_titles is not None:
            self.set_suplotgrid_titles(grid_axis=GridAxis.ROWS)(row_titles)

        return self
