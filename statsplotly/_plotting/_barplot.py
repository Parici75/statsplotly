"""Bar plots"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio

from statsplotly import constants
from statsplotly.exceptions import StatsPlotSpecificationError
from statsplotly.plot_objects.layout import BarLayout
from statsplotly.plot_objects.trace import BarTrace

# Specifiers
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.plot_specifiers.data import (
    AggregationSpecifier,
    AggregationTraceData,
    DataDimension,
    DataHandler,
    DataPointer,
    TraceData,
)
from statsplotly.plot_specifiers.figure import create_fig
from statsplotly.plot_specifiers.layout import (
    AxesSpecifier,
    ColoraxisReference,
    LegendSpecifier,
)

# Trace objects
from statsplotly.plot_specifiers.trace import OrientedPlotSpecifier

pio.templates.default = constants.DEFAULT_TEMPLATE
np.seterr(invalid="ignore")

logger = logging.getLogger(__name__)


def barplot(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    orientation: str | None = None,
    slicer: str | None = None,
    slice_order: list[str] | None = None,
    color: str | None = None,
    color_palette: list[str] | str | None = None,
    shared_coloraxis: bool = False,
    color_limits: Sequence[float] | None = None,
    logscale: float | None = None,
    colorbar: bool = True,
    text: str | None = None,
    axis: str | None = None,
    opacity: float | None = None,
    barmode: str | None = None,
    error_bar: str | None = None,
    aggregation_func: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    x_range: Sequence[float | str] | None = None,
    y_range: Sequence[float | str] | None = None,
    fig: go.Figure = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Draws a barplot across levels of categorical variable.

    Args:
        data: A :obj:`pandas.DataFrame`
        x: The name of the `x` dimension column in `data`.
        y: The name of the `y` dimension column in `data`.
        orientation: A :obj:`~statsplotly.plot_specifiers.trace.PlotOrientation` value to force the orientation of the plot.
        slicer: The name of the column in `data` with values to slice the data : one trace is drawn for each level of the `slicer` dimension.
        slice_order: A list of identifiers to order and/or subset data slices specified by `slicer`.
        color: The name of the column in `data` with values to map onto the colormap.
        color_palette:
            - A string refering to a built-in `plotly`, `seaborn` or `matplotlib` colormap.
            - A list of CSS color names or HTML color codes.

            The color palette is used, by order of precedence :
                - To map color data specified by the `color` parameter onto the corresponding colormap.
                - To assign discrete colors to `slices` of data.

        shared_coloraxis: If True, colorscale limits are shared across slices of data.
        color_limits: A tuple specifying the (min, max) values of the colormap.
        logscale: A float specifying the log base to use for colorscaling.
        colorbar: If True, draws a colorbar.
        text: A string or the name of the column in `data` with values to appear in the hover tooltip. Column names can be concatenated with '+' to display values from multiple columns. Ignored when `aggregation_func` is not None.
        axis: A :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval to specify bar opacity.
        barmode: A :obj:`~statsplotly.plot_specifiers.layout.BarMode` value.
        error_bar: A :obj:`~statsplotly.plot_specifiers.data.ErrorBarType` value or a `Callable` taking the `x` or `y` dimension as input and returning a (inferior_limit, superior_limit) tuple.
        aggregation_func: A :obj:`~statsplotly.plot_specifiers.data.AggregationType` value or a `Callable` taking the `x` or `y` dimension as input and returning a single value.
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

    if color is not None and aggregation_func is not None:
        raise StatsPlotSpecificationError("Color coding can not be used with aggregation")

    data_handler = DataHandler.build_handler(
        data=data,
        data_pointer=DataPointer(x=x, y=y, slicer=slicer, color=color, text=text),
        slice_order=slice_order,
    )

    plot_specifier = OrientedPlotSpecifier(
        prefered_orientation=orientation, data_types=data_handler.data_types
    )

    aggregation_specifier = AggregationSpecifier(
        aggregation_func=aggregation_func,
        aggregated_dimension=plot_specifier.anchored_dimension or plot_specifier.anchor_dimension,
        error_bar=error_bar,
        data_pointer=data_handler.data_pointer,
        data_types=data_handler.data_types,
    )

    color_specifier = ColorSpecifier.build_from_color_data(
        color_data=data_handler.get_data("color"),
        coloraxis_reference=ColoraxisReference.MAIN_COLORAXIS,
        color_palette=color_palette,
        logscale=logscale,
        colorbar=colorbar,
        color_limits=color_limits,
        opacity=opacity,
    )

    traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}
    traces_data: list[TraceData] = []
    for (slice_name, slice_data), trace_color in zip(
        data_handler.iter_slices(),
        color_specifier.get_color_hues(n_colors=data_handler.n_slices),
        strict=True,
    ):
        trace_data: AggregationTraceData | TraceData
        if aggregation_specifier.aggregation_func is not None:
            trace_data = AggregationTraceData.build_aggregation_trace_data(
                data=slice_data,
                aggregation_specifier=aggregation_specifier,
            )
        else:
            trace_data = TraceData.build_trace_data(
                data=slice_data, pointer=data_handler.data_pointer
            )

        traces[slice_name] = BarTrace.build_trace(
            trace_data=trace_data,
            trace_name=slice_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            bar_plot_specifier=plot_specifier,
        ).to_plotly_trace()

        traces_data.append(trace_data)

    legend_specifier = LegendSpecifier(
        data_pointer=data_handler.data_pointer,
        title=title,
        x_label=x_label,
        y_label=y_label,
        x_transformation=(
            aggregation_func
            if aggregation_specifier.aggregation_plot_dimension is DataDimension.X
            and isinstance(aggregation_func, str)
            else None
        ),
        y_transformation=(
            aggregation_func
            if aggregation_specifier.aggregation_plot_dimension is DataDimension.Y
            and isinstance(aggregation_func, str)
            else None
        ),
        error_bar=error_bar if isinstance(error_bar, str) else None,
    )

    axes_specifier = AxesSpecifier(
        traces=traces_data,
        axis_format=axis,
        legend=legend_specifier,
        x_range=x_range,
        y_range=y_range,
    )

    coloraxis = color_specifier.build_coloraxis(
        color_data=data_handler.get_data("color"), shared=shared_coloraxis
    )

    layout = BarLayout.build_layout(
        axes_specifier=axes_specifier, coloraxis=coloraxis, barmode=barmode
    )

    # Create fig
    fig = create_fig(fig=fig, traces=traces, layout=layout, row=row, col=col)

    return fig
