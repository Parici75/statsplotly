from __future__ import annotations

import logging
import re
from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from statsplotly.plot_objects.layout import SceneLayout, layout_type
from statsplotly.plot_objects.trace import BaseTrace
from statsplotly.plot_specifiers.data import DataDimension
from statsplotly.plot_specifiers.layout import ColoraxisReference
from statsplotly.plot_specifiers.trace import HistogramSpecifier, JointplotSpecifier

from ._utils import FigureSubplotFormatter, SharedGridAxis, SubplotGridFormatter

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BasePlot(FigureSubplotFormatter, Generic[T], metaclass=ABCMeta):
    plot_specifier: T

    @abstractmethod
    def initialize(
        cls, fig: go.Figure | None, row: int | None, col: int | None, plot_specifier: T
    ) -> BasePlot[T]:
        """This method implements the logic to initialize a subplot layout."""

    @property
    @abstractmethod
    def main_row(self) -> int:
        """This method returns the index of the main row in the subplot (i.e., the row with the core Figure object)."""


class HistogramPlot(BasePlot[HistogramSpecifier]):
    plot_specifier: HistogramSpecifier

    @property
    def central_tendency_col(self) -> int:
        return self.col + 1 if self.plot_specifier.dimension is DataDimension.Y else self.col

    @property
    def main_row(self) -> int:
        return (
            self.row + 1
            if (
                self.plot_specifier.central_tendency is not None
                and self.plot_specifier.dimension is DataDimension.X
            )
            else self.row
        )

    @classmethod
    def initialize(
        cls,
        fig: go.Figure | None,
        row: int | None,
        col: int | None,
        plot_specifier: HistogramSpecifier,
    ) -> HistogramPlot:
        if fig is None:
            if plot_specifier.central_tendency is not None:
                match plot_specifier.dimension:
                    case DataDimension.X:
                        fig = make_subplots(
                            rows=2,
                            cols=1,
                            row_heights=[0.2, 0.8],
                            vertical_spacing=0.05,
                            shared_xaxes=True,
                        )
                    case DataDimension.Y:
                        fig = make_subplots(
                            rows=1,
                            cols=2,
                            column_widths=[0.8, 0.2],
                            horizontal_spacing=0.05,
                            shared_yaxes=True,
                        )
            else:
                fig = make_subplots(rows=1, cols=1)

        return cls(
            fig=fig,
            row=row,
            col=col,
            plot_specifier=plot_specifier,
        )


class JointplotPlot(BasePlot[JointplotSpecifier]):
    plot_specifier: JointplotSpecifier

    @property
    def main_row(self) -> int:
        return self.row + 1 if self.plot_specifier.plot_x_distribution else self.row

    def tidy_plot(self) -> JointplotPlot:
        # We are limited to formatting only axes in the figure, as in the current implementation of jointplot,
        # coloraxes are managed at the trace level instead of the layout level.
        # TODO: Define coloraxes at the layout level and use _SubplotGridCommonColoraxisFormatter class for formatting.
        subplot_grid_formatter = SubplotGridFormatter(fig=self.fig)
        if self.plot_specifier.plot_x_distribution:
            subplot_grid_formatter.set_common_axis_limit(
                shared_grid_axis=SharedGridAxis.COLS, common_range=True, link_axes=True
            )
        if self.plot_specifier.plot_y_distribution:
            subplot_grid_formatter.set_common_axis_limit(
                shared_grid_axis=SharedGridAxis.ROWS, common_range=True, link_axes=True
            )
        subplot_grid_formatter.tidy_subplots()

        return self

    @classmethod
    def initialize(
        cls,
        fig: go.Figure | None,
        row: int | None,
        col: int | None,
        plot_specifier: JointplotSpecifier,
    ) -> JointplotPlot:
        if fig is None:
            fig = make_subplots(
                rows=2 if plot_specifier.plot_x_distribution else 1,
                cols=2 if plot_specifier.plot_y_distribution else 1,
                row_heights=[0.2, 0.8] if plot_specifier.plot_x_distribution else [1],
                column_widths=[0.8, 0.2] if plot_specifier.plot_y_distribution else [1],
                vertical_spacing=0.05,
                horizontal_spacing=0.05,
                shared_xaxes=True if plot_specifier.plot_x_distribution else False,
                shared_yaxes=True if plot_specifier.plot_y_distribution else False,
            )

        return cls(
            fig=fig,
            row=row,
            col=col,
            plot_specifier=plot_specifier,
        )


def create_fig(  # noqa: PLR0912 C901
    fig: go.Figure,
    traces: dict[str, BaseTrace],
    layout: layout_type,
    row: int | None,
    col: int | None,
    secondary_y: bool = False,
) -> go.Figure:
    """Creates or updates a figure with the appropriate layout."""
    layout_dict = layout.model_dump()
    _row = row if row is not None else 1
    _col = col if col is not None else 1
    if fig is None:
        if row is None and col is None:
            return go.Figure(
                data=list(traces.values()),
                layout=go.Layout(layout_dict),
            )

        fig = make_subplots(rows=_row, cols=_col)

    figure_subplot_formatter = FigureSubplotFormatter(fig=fig, row=_row, col=_col)

    if (current_coloraxis := layout_dict.get(ColoraxisReference.MAIN_COLORAXIS)) is not None:
        if current_coloraxis.get("colorscale") is not None:
            # Arange colorbar layout
            figure_subplot_formatter.format_colorbar_in_subplots_layout(
                coloraxis_dict=current_coloraxis,
            )

            # Update coloraxis reference
            figure_subplot_formatter.update_coloraxis_reference(
                coloraxis_reference=ColoraxisReference.MAIN_COLORAXIS,
                layout_dict=layout_dict,
                traces=traces,
            )

    # Add the new traces
    for trace in traces.values():
        fig.add_trace(trace, row=_row, col=_col, secondary_y=secondary_y)

    # Rename layout axes keys to match position in the layout
    if isinstance(layout, SceneLayout):
        scene = fig._grid_ref[_row - 1][_col - 1][0][1][0]
        layout_dict[scene] = layout_dict.pop("scene")

    else:
        # 2D plot
        axis = fig._grid_ref[_row - 1][_col - 1]
        if secondary_y:
            xaxis_ref, yaxis_ref = axis[1][1]
        else:
            # Extract xaxis and yaxis axes
            xaxis_ref, yaxis_ref = axis[0][1]
        # Update layout
        layout_dict[xaxis_ref] = layout_dict.pop("xaxis")
        layout_dict[yaxis_ref] = layout_dict.pop("yaxis")
        # Rename axes references
        for axis_ref in [xaxis_ref, yaxis_ref]:
            if (axis_number_pattern := re.search(r"\d+", axis_ref)) is not None:
                axis_number = axis_number_pattern.group()
                if (scaleanchor := layout_dict[axis_ref].get("scaleanchor")) is not None:
                    scaleanchor_root = re.sub(r"\d", axis_number_pattern.group(), scaleanchor)
                    layout_dict[axis_ref].update(
                        {"scaleanchor": f"{scaleanchor_root}{axis_number}"}
                    )

        # Remove axes titles when shared axes
        if _row < len(fig._grid_ref) and fig.layout[xaxis_ref].matches is not None:
            layout_dict[xaxis_ref]["title"] = None
        if _col > 1 and fig.layout[yaxis_ref].matches is not None:
            layout_dict[yaxis_ref]["title"] = None

    # Tidy up legend
    figure_subplot_formatter.hide_legend_duplicates()

    # Update layout
    fig.update_layout({key: value for key, value in layout_dict.items() if value is not None})

    return fig
