"""Heatmap plots"""

import logging
from collections.abc import Sequence
from functools import reduce

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio

from statsplotly import constants
from statsplotly.plot_objects.layout import HeatmapLayout
from statsplotly.plot_objects.trace import HeatmapTrace

# Specifiers
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.plot_specifiers.data import (
    DataDimension,
    DataHandler,
    DataPointer,
    DataProcessor,
    SliceTraceType,
    TraceData,
)
from statsplotly.plot_specifiers.figure import create_fig
from statsplotly.plot_specifiers.layout import (
    AxesSpecifier,
    AxisType,
    ColoraxisReference,
    LegendSpecifier,
    add_update_menu,
)

pio.templates.default = constants.DEFAULT_TEMPLATE
np.seterr(invalid="ignore")

logger = logging.getLogger(__name__)


def heatmap(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    slicer: str | None = None,
    slice_order: list[str] | None = None,
    color_palette: list[str] | str | None = None,
    shared_coloraxis: bool = False,
    color_limits: Sequence[float] | None = None,
    logscale: float | None = None,
    colorbar: bool = True,
    text: str | None = None,
    axis: str | None = None,
    opacity: float | None = None,
    normalizer: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    z_label: str | None = None,
    title: str | None = None,
    x_range: Sequence[float | str] | None = None,
    y_range: Sequence[float | str] | None = None,
    fig: go.Figure = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Draws a heatmap.

    Args:
        data: A :obj:`pandas.DataFrame`
        x: The name of the `x` dimension column in `data`.
        y: The name of the `y` dimension column in `data`.
        z: The name of the `z` dimension (i.e., color) column in `data`.
        slicer: The name of the column in `data` with values to slice the data : one trace is drawn for each level of the `slicer` dimension.
        slice_order: A list of identifiers to order and/or subset data slices specified by `slicer`.
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
        text: A string or the name of the column in `data` with values to appear in the hover tooltip. Column names can be concatenated with '+' to display values from multiple columns.
        axis: A :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval to specify heatmap opacity.
        normalizer: The normalizer for the `z` dimension. A :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        x_label: A string to label the x_axis in place of the corresponding column name in `data`.
        y_label: A string to label the y_axis in place of the corresponding column name in `data`.
        z_label: A string to label the coloraxis in place of the corresponding column name in `data`.
        title: A string to label the resulting plot.
        x_range: A tuple defining the (min_range, max_range) of the x_axis.
        y_range: A tuple defining the (min_range, max_range) of the y_axis.
        fig: A :obj:`plotly.graph_obj.Figure` to add the plot to. Use in conjunction with row and col.
        row: An integer identifying the row to add the plot to.
        col: An integer identifying the colum to add the plot to.

    Returns:
        A :obj:`plotly.graph_obj.Figure`.
    """

    data_handler = DataHandler.build_handler(
        data=data,
        data_pointer=DataPointer(x=x, y=y, z=z, slicer=slicer, text=text),
        slice_order=slice_order,
    )

    color_specifier = ColorSpecifier.build_from_color_data(
        color_data=data_handler.get_data(DataDimension.Z),
        coloraxis_reference=ColoraxisReference.MAIN_COLORAXIS,
        color_palette=color_palette,
        logscale=logscale,
        color_limits=color_limits,
        colorbar=colorbar,
        opacity=opacity,
    )

    data_processor = DataProcessor(normalizer={DataDimension.Z: normalizer})

    traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}
    traces_data: list[TraceData] = []

    if data_handler.n_slices > 1 and not all(
        len(
            reduce(
                np.intersect1d,
                data_handler.data.groupby(data_handler.data_pointer.slicer)[
                    getattr(data_handler.data_pointer, dimension)
                ].unique(),
            )
        )
        > 0
        for dimension in DataDimension
    ):
        global_trace = HeatmapTrace.build_trace(
            trace_data=TraceData.build_trace_data(
                data=data_handler.data,
                pointer=data_handler.data_pointer,
            ),
            trace_name=SliceTraceType.ALL_DATA,
            color_specifier=color_specifier,
        )
        traces[global_trace.name] = global_trace.to_plotly_trace()

    for slice_name, slice_data in data_handler.iter_slices():
        trace_data = TraceData.build_trace_data(
            data=slice_data,
            pointer=data_handler.data_pointer,
            processor=data_processor,
        )

        traces[slice_name] = HeatmapTrace.build_trace(
            trace_data=trace_data,
            trace_name=slice_name,
            color_specifier=color_specifier,
        ).to_plotly_trace()
        traces_data.append(trace_data)

    legend_specifier = LegendSpecifier(
        data_pointer=data_handler.data_pointer,
        title=title,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
        axis_type=AxisType.TWO_DIMENSIONAL,
    )

    axes_specifier = AxesSpecifier(
        traces=traces_data,
        axis_format=axis,
        legend=legend_specifier,
        x_range=x_range,
        y_range=y_range,
    )

    coloraxis = color_specifier.build_coloraxis(
        color_data=data_handler.get_data(DataDimension.Z), shared=shared_coloraxis
    )

    layout = HeatmapLayout.build_layout(
        axes_specifier=axes_specifier,
        coloraxis=coloraxis,
    )

    if fig is not None:
        preplotted_traces = {trace.name: trace for trace in fig.data}
    else:
        preplotted_traces = {}

    # Create fig
    fig = create_fig(fig=fig, traces=traces, layout=layout, row=row, col=col)

    # Add menus
    if data_handler.n_slices > 1:
        fig = add_update_menu(
            fig=fig, data_handler=data_handler, preplotted_traces=preplotted_traces
        )

    return fig
