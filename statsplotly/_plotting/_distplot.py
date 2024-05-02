"""Distribution plots"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio

from statsplotly import constants
from statsplotly.plot_objects.layout import HistogramLayout
from statsplotly.plot_objects.trace import HistogramLineTrace

# Specifiers
from statsplotly.plot_specifiers.color import HistogramColorSpecifier
from statsplotly.plot_specifiers.data import (
    AGG_TO_ERROR_MAPPING,
    CentralTendencyType,
    DataDimension,
    DataHandler,
    DataPointer,
    TraceData,
)
from statsplotly.plot_specifiers.figure import HistogramPlot, create_fig
from statsplotly.plot_specifiers.layout import AxesSpecifier, LegendSpecifier

# Trace objects
from statsplotly.plot_specifiers.trace import HistogramSpecifier, TraceMode

# Helpers
from ._plot import plot
from .helpers import plot_distplot_traces

pio.templates.default = constants.DEFAULT_TEMPLATE
np.seterr(invalid="ignore")

logger = logging.getLogger(__name__)


def distplot(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    slicer: str | None = None,
    slice_order: list[str] | None = None,
    color_palette: list[str] | str | None = None,
    axis: str | None = None,
    opacity: float | None = None,
    hist: bool = True,
    rug: bool | None = None,
    ecdf: bool | None = None,
    kde: bool | None = None,
    step: bool | None = None,
    equal_bins: bool | None = None,
    bins: Sequence[float] | int | str | None = None,
    cumulative: bool | None = None,
    histnorm: str | None = None,
    central_tendency: str | None = None,
    vlines: dict[str, tuple[str, float]] | None = None,
    hlines: dict[str, tuple[str, float]] | None = None,
    barmode: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    x_range: Sequence[float] | None = None,
    y_range: Sequence[float] | None = None,
    fig: go.Figure = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Draws the distribution of x (vertical histogram) or y (horizontal histograms) values.

    Args:
        data: A :obj:`pandas.DataFrame`
        x: The name of the `x` dimension column in `data`. If not None, draws vertical histograms.
        y: The name of the `y` dimension column in `data`. If not None, draws horizontal histograms.
        slicer: The name of the column in `data` with values to slice the data : one trace is drawn for each level of the `slicer` dimension.
        slice_order: A list of identifiers to order and/or subset data slices specified by `slicer`.
        color_palette:
            - A string refering to a built-in `plotly`, `seaborn` or `matplotlib` colormap.
            - A list of CSS color names or HTML color codes.

            The color palette is used to assign discrete colors to `slices` of data.
        axis: A :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval to specify bar and line opacity.
        hist: If True, plot histogram bars.
        rug: If True, plot rug bars of the underlying data.
        ecdf: If True, plot the Empirical Cumulative Density Function.
        kde: If True, plot a line of a Kernel Density Estimation of the distribution.
        step: If True, plot a step histogram instead of a standard histogram bars.
        equal_bins: If True, uses the same bins for all `slices` in the data.
        bins: A string, integer, or sequence specifying the `bins` parameter for :func:`numpy.histogram`.
        cumulative: If True, draws a cumulative histogram.
        histnorm: A :obj:`~statsplotly.plot_specifiers.data.HistogramNormType` value.
        central_tendency: A :obj:`~statsplotly.plot_specifiers.data.CentralTendencyType` value.
        vlines: A dictionary of {slice: (line_name, vertical_coordinates)} to draw vertical lines.
        hlines: A dictionary of {slice: (line_name, horizontal_coordinates)} to draw horizontal lines.
        barmode: A :obj:`~statsplotly.plot_specifiers.layout.HistogramBarMode` value.
        x_label: A string to label the x_axis in place of the corresponding column name in `data`.
        y_label: A string to label the y_axis in place of the corresponding column name in `data`.
        title: A string for the title of the plot.
        x_range: A tuple defining the (min_range, max_range) of the x_axis.
        y_range: A tuple defining the (min_range, max_range) of the y_axis.
        fig: A :obj:`plotly.graph_obj.Figure` to add the plot to. Use in conjunction with row and col.
        row: An integer identifying the row to add the plot to.
        col: An integer identifying the column to add the plot to.

    Returns:
        A :obj:`plotly.graph_obj.Figure`.
    """

    data_handler = DataHandler.build_handler(
        data=data,
        data_pointer=DataPointer(x=x, y=y, slicer=slicer),
        slice_order=slice_order,
    )

    histogram_dimension = (
        DataDimension.X if data_handler.data_pointer.x is not None else DataDimension.Y
    )

    histogram_specifier = HistogramSpecifier(
        hist=hist,
        rug=rug,
        ecdf=ecdf,
        kde=kde,
        step=step,
        bins=bins,
        cumulative=cumulative,
        histnorm=histnorm,
        central_tendency=central_tendency,
        data_type=getattr(data_handler.data_types, histogram_dimension),
        dimension=histogram_dimension,
    )
    if equal_bins:
        # Call histogram on all data to set bin edge attribute
        histogram_specifier.bin_edges = histogram_specifier.get_histogram_bin_edges(
            data_handler.get_data(histogram_dimension)
        )[0]

    color_specifier = HistogramColorSpecifier(
        color_palette=color_palette, opacity=opacity, barmode=barmode
    )

    traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}
    traces_data: list[TraceData] = []
    for (slice_name, slice_data), trace_color in zip(
        data_handler.iter_slices(),
        color_specifier.get_color_hues(n_colors=data_handler.n_slices),
        strict=True,
    ):
        trace_data = TraceData.build_trace_data(data=slice_data, pointer=data_handler.data_pointer)

        traces.update(
            plot_distplot_traces(
                trace_data=trace_data,
                trace_name=slice_name,
                trace_color=trace_color,
                color_specifier=color_specifier,
                histogram_specifier=histogram_specifier,
            )
        )

        if vlines is not None:
            if (vline := vlines.get(slice_name)) is not None:
                line_trace = HistogramLineTrace.build_trace(
                    vline_coordinates=vline,
                    trace_data=trace_data,
                    trace_name=slice_name,
                    trace_color=trace_color,
                    histogram_specifier=histogram_specifier,
                )
                traces[line_trace.name] = line_trace.to_plotly_trace()

        if hlines is not None:
            if (hline := hlines.get(slice_name)) is not None:
                line_trace = HistogramLineTrace.build_trace(
                    hline_coordinates=hline,
                    trace_data=trace_data,
                    trace_name=slice_name,
                    trace_color=trace_color,
                    histogram_specifier=histogram_specifier,
                )
                traces[line_trace.name] = line_trace.to_plotly_trace()

        traces_data.append(trace_data)

    legend_specifier = LegendSpecifier(
        data_pointer=data_handler.data_pointer,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_transformation=(
            histogram_specifier.histnorm if histogram_dimension is DataDimension.X else None
        ),
        x_transformation=(
            histogram_specifier.histnorm if histogram_dimension is DataDimension.Y else None
        ),
    )

    axes_specifier = AxesSpecifier(
        traces=traces_data,
        axis_format=axis,
        legend=legend_specifier,
        x_range=x_range,
        y_range=y_range,
    )

    layout = HistogramLayout.build_layout(axes_specifier=axes_specifier, barmode=barmode)
    figure_plot = HistogramPlot.initialize(
        plot_specifier=histogram_specifier, fig=fig, row=row, col=col
    )

    if histogram_specifier.central_tendency is not None:
        if histogram_specifier.central_tendency is CentralTendencyType.MEAN:
            central_tendency_data = data_handler.get_mean(histogram_dimension)
        elif histogram_specifier.central_tendency is CentralTendencyType.MEDIAN:
            central_tendency_data = data_handler.get_median(histogram_dimension)
        else:
            raise ValueError(
                "Unsupported parameter for distribution central tendency:"
                f" {histogram_specifier.central_tendency.value}"
            )

        fig = plot(
            fig=figure_plot.fig,
            row=figure_plot.row,
            col=figure_plot.central_tendency_col,
            data=central_tendency_data,
            x=(
                histogram_specifier.central_tendency
                if histogram_specifier.dimension is DataDimension.X
                else (slicer or "index")
            ),
            y=(
                histogram_specifier.central_tendency
                if histogram_specifier.dimension is DataDimension.Y
                else (slicer or "index")
            ),
            slicer=slicer,
            mode=TraceMode.MARKERS,
            color_palette=color_palette,
            error_x=(
                AGG_TO_ERROR_MAPPING[histogram_specifier.central_tendency]
                if histogram_specifier.dimension is DataDimension.X
                else None
            ),
            error_y=(
                None
                if histogram_specifier.dimension is DataDimension.X
                else AGG_TO_ERROR_MAPPING[histogram_specifier.central_tendency]
            ),
        )
        # Update name
        fig.for_each_trace(
            lambda trace: trace.update(
                name=(
                    f"{trace.name} {histogram_specifier.central_tendency.value} +/-"
                    f" {AGG_TO_ERROR_MAPPING[histogram_specifier.central_tendency].value}"
                ),
            ),
            row=figure_plot.row,
            col=figure_plot.central_tendency_col,
        )
        # Hide axes
        axis_idx = (
            str(figure_plot.row * figure_plot.central_tendency_col)
            if figure_plot.row * figure_plot.central_tendency_col > 1
            else ""
        )
        fig.update_layout(
            {
                f"xaxis{axis_idx}": {"visible": False},
                f"yaxis{axis_idx}": {"visible": False},
            }
        )

    # Create fig
    fig = create_fig(
        fig=figure_plot.fig,
        traces=traces,
        layout=layout,
        row=figure_plot.main_row,
        col=figure_plot.col,
    )

    return fig
