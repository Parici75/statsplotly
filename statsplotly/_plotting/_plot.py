"""Line or scatter plots"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio

from statsplotly import constants
from statsplotly.exceptions import StatsPlotSpecificationError
from statsplotly.plot_objects.layout import ScatterLayout, SceneLayout
from statsplotly.plot_objects.trace import Scatter3DTrace, ScatterTrace

# Specifiers
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.plot_specifiers.data import (
    DataDimension,
    DataHandler,
    DataPointer,
    DataProcessor,
    TraceData,
)
from statsplotly.plot_specifiers.figure import create_fig
from statsplotly.plot_specifiers.layout import (
    AxesSpecifier,
    AxisFormat,
    ColoraxisReference,
    LegendSpecifier,
)

# Trace objects
from statsplotly.plot_specifiers.trace import ScatterSpecifier, TraceMode

# Helpers
from .helpers import plot_scatter_traces

pio.templates.default = constants.DEFAULT_TEMPLATE
np.seterr(invalid="ignore")

logger = logging.getLogger(__name__)


def plot(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    slicer: str | None = None,
    slice_order: list[str] | None = None,
    color: str | None = None,
    color_palette: list[str] | str | None = None,
    shared_coloraxis: bool = False,
    color_limits: Sequence[float] | None = None,
    logscale: float | None = None,
    colorbar: bool = True,
    text: str | None = None,
    marker: str | None = None,
    mode: str | None = None,
    axis: str | None = None,
    opacity: str | float | None = None,
    jitter_x: float = 0,
    jitter_y: float = 0,
    jitter_z: float = 0,
    normalizer_x: str | None = None,
    normalizer_y: str | None = None,
    normalizer_z: str | None = None,
    shaded_error: str | None = None,
    error_x: str | None = None,
    error_y: str | None = None,
    error_z: str | None = None,
    fit: str | None = None,
    size: float | str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    z_label: str | None = None,
    title: str | None = None,
    x_range: Sequence[float | str] | None = None,
    y_range: Sequence[float | str] | None = None,
    z_range: Sequence[float | str] | None = None,
    fig: go.Figure = None,
    row: int | None = None,
    col: int | None = None,
    secondary_y: bool = False,
) -> go.Figure:
    """Draws a line/scatter plot across levels of a categorical variable.

    Args:
        data: A :obj:`pandas.DataFrame`
        x: The name of the `x` dimension column in `data`.
        y: The name of the `y` dimension column in `data`.
        z: The name of the `z` dimension column in `data`.
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
        text: A string or the name of the column in `data` with values to appear in the hover tooltip. Column names can be concatenated with '+' to display values from multiple columns.
        marker: A valid marker symbol or the name of the column in `data` with values to assign marker symbols.
        mode: A :obj:`~statsplotly.plot_specifiers.trace.TraceMode` value.
        axis: A :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval or the name of the column in `data` with values to specify marker opacity.
        jitter_x: A numeric value to specify jitter amount on the `x` dimension.
        jitter_y: A numeric value to specify jitter amount on the `y` dimension.
        jitter_z: A numeric value to specify jitter amount on the `z` dimension.
        normalizer_x: The normalizer for the `x` dimension. A :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        normalizer_y: The normalizer for the `y` dimension. A :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        normalizer_z: The normalizer for the `z` dimension. A :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        shaded_error: The name of the column in `data` with values to plot continuous error shade.
        error_x: The name of the column in `data` with values to plot error bar in the `x` dimension.
        error_y: The name of the column in `data` with values to plot error bar in the `y` dimension.
        error_z: The name of the column in `data` with values to plot error bar in the `z` dimension.
        fit: A :obj:`~statsplotly.plot_specifiers.data.RegressionType` value. Computes and plot the corresponding regression.
        size: A numeric value or the name of the column in `data` with values to assign mark sizes.
        x_label: A string to label the x_axis in place of the corresponding column name in `data`.
        y_label: A string to label the y_axis in place of the corresponding column name in `data`.
        z_label: A string to label the z_axis in place of the corresponding column name in `data`.
        title: A string for the title of the plot.
        x_range: A tuple defining the (min_range, max_range) of the x_axis.
        y_range: A tuple defining the (min_range, max_range) of the y_axis.
        z_range: A tuple defining the (min_range, max_range) of the z_axis.
        fig: A :obj:`plotly.graph_obj.Figure` to add the plot to. Use in conjunction with row and col.
        row: An integer identifying the row to add the plot to.
        col: An integer identifying the column to add the plot to.
        secondary_y: If True, plot on a secondary y_axis of the `fig` object.

    Returns:
        A :obj:`plotly.graph_obj.Figure`.
    """

    if (color is not None or size is not None or marker is not None) and mode is None:
        mode = TraceMode.MARKERS
    if color is not None and mode is TraceMode.LINES:
        raise ValueError("Only markers can be mapped to colormap")
    if size is not None and mode is TraceMode.LINES:
        raise ValueError("Size specification only applies to markers")
    if z is not None:
        if mode is None:
            mode = TraceMode.MARKERS
        if fit is not None:
            raise ValueError("Regression can not be computed on a three-dimensional plot")
        if size is None:
            size = constants.DEFAULT_MARKER_SIZE

    data_handler = DataHandler.build_handler(
        data=data,
        data_pointer=DataPointer(
            x=x,
            y=y,
            z=z,
            slicer=slicer,
            shaded_error=shaded_error,
            error_x=error_x,
            error_y=error_y,
            error_z=error_z,
            color=color,
            text=text,
            marker=marker,
            size=size,
            opacity=opacity,
        ),
        slice_order=slice_order,
    )

    scatter_specifier = ScatterSpecifier(
        mode=mode, regression_type=fit, data_types=data_handler.data_types
    )

    if opacity is None and scatter_specifier.regression_type is not None:
        logger.debug(
            f"Regression plot is on, setting opacity to {constants.DEFAULT_TRANSPARENCE_OPACITY}"
        )
        opacity = constants.DEFAULT_TRANSPARENCE_OPACITY

    legend_specifier = LegendSpecifier(
        data_pointer=data_handler.data_pointer,
        shaded_error=shaded_error,
        title=title,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
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

    data_processor = DataProcessor(
        jitter_settings={
            DataDimension.X: jitter_x,
            DataDimension.Y: jitter_y,
            DataDimension.Z: jitter_z,
        },
        normalizer={
            DataDimension.X: normalizer_x,
            DataDimension.Y: normalizer_y,
            DataDimension.Z: normalizer_z,
        },
    )

    traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}
    traces_data: list[TraceData] = []
    for (slice_name, slice_data), trace_color in zip(
        data_handler.iter_slices(),
        color_specifier.get_color_hues(n_colors=data_handler.n_slices),
        strict=True,
    ):
        trace_data = TraceData.build_trace_data(
            data=slice_data,
            pointer=data_handler.data_pointer,
            processor=data_processor,
        )

        if data_handler.data_pointer.z is not None:
            traces[slice_name] = Scatter3DTrace.build_trace(
                trace_data=trace_data,
                trace_name=slice_name,
                trace_color=trace_color,
                color_specifier=color_specifier,
                mode=scatter_specifier.mode,
            ).to_plotly_trace()

        else:
            traces.update(
                plot_scatter_traces(
                    trace_data=trace_data,
                    trace_name=slice_name,
                    trace_color=trace_color,
                    color_specifier=color_specifier,
                    scatter_specifier=scatter_specifier,
                )
            )
        traces_data.append(trace_data)

    axes_specifier = AxesSpecifier(
        traces=traces_data,
        axis_format=axis,
        legend=legend_specifier,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
    )
    if axes_specifier.axis_format is AxisFormat.ID_LINE:
        if data_handler.data_pointer.z is not None:
            raise StatsPlotSpecificationError(
                f"axis={axes_specifier.axis_format.value} is not compatible with three-dimensional"
                " plotting"
            )
        traces["id_line"] = ScatterTrace.build_id_line(
            x_values=data_handler.get_data("x"),
            y_values=data_handler.get_data("y"),
        ).to_plotly_trace()

    coloraxis = color_specifier.build_coloraxis(
        color_data=data_handler.get_data("color"), shared=shared_coloraxis
    )

    layout_constructor: Any
    if data_handler.data_pointer.z is not None:
        layout_constructor = SceneLayout
    else:
        layout_constructor = ScatterLayout
    layout = layout_constructor.build_layout(axes_specifier=axes_specifier, coloraxis=coloraxis)

    # Create fig
    fig = create_fig(
        fig=fig,
        traces=traces,
        layout=layout,
        row=row,
        col=col,
        secondary_y=secondary_y,
    )

    return fig
