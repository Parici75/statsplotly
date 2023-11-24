"""Main module for plotting functions."""

import logging
from collections.abc import Sequence
from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from pandas.api.types import is_numeric_dtype, is_object_dtype
from plotly.subplots import make_subplots

from statsplotly import constants
from statsplotly.exceptions import StatsPlotSpecificationError
from statsplotly.plot_objects.layout_objects import (
    BarLayout,
    CategoricalLayout,
    HeatmapLayout,
    HistogramLayout,
    ScatterLayout,
    SceneLayout,
)
from statsplotly.plot_objects.trace_objects import (
    BarTrace,
    BaseTrace,
    BoxTrace,
    HeatmapTrace,
    HistogramLineTrace,
    Scatter3DTrace,
    ScatterTrace,
    StripTrace,
    ViolinTrace,
)

# Specifiers
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.plot_specifiers.data import (
    AGG_TO_ERROR_MAPPING,
    AggregationSpecifier,
    AggregationTraceData,
    CentralTendencyType,
    DataDimension,
    DataHandler,
    DataPointer,
    DataProcessor,
    SliceTraceType,
    TraceData,
)
from statsplotly.plot_specifiers.layout import (
    AxesSpecifier,
    AxisFormat,
    ColoraxisReference,
    LegendSpecifier,
)

# Trace objects
from statsplotly.plot_specifiers.trace import (
    CategoricalPlotType,
    HistogramSpecifier,
    JointplotSpecifier,
    JointplotType,
    MarginalPlotDimension,
    ScatterSpecifier,
    TraceMode,
)

# Helpers
from statsplotly.plotting.helpers import (
    plot_jointplot_main_traces,
    plot_marginal_traces,
    plot_scatter_traces,
)
from statsplotly.utils.figure_utils import create_fig
from statsplotly.utils.layout_utils import add_update_menu, adjust_jointplot_legends

pio.templates.default = constants.DEFAULT_TEMPLATE
np.seterr(invalid="ignore")

logger = logging.getLogger(__name__)


# Simple line or scatter plot
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
    size: float | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    z_label: str | None = None,
    title: str | None = None,
    x_range: Sequence[float | str] | None = None,
    y_range: Sequence[float | str] | None = None,
    z_range: Sequence[float | str] | None = None,
    fig: go.Figure = None,
    row: int = 1,
    col: int = 1,
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
                - To map color data of the `color` parameter onto the corresponding colormap.
                - To assign discrete colors to `slices` of data.
        shared_coloraxis: If True, colorscale is shared across slices of data.
        color_limits: A tuple specifying the (min, max) values of the colormap.
        logscale: A float specifying the log base to use for colorscaling.
        colorbar: If True, draws a colorbar.
        text: A string or the name of the column in `data` with values to appear in the hover tooltip. Column names can be concatenated with '+' to display values from multiple columns.
        marker: A valid marker symbol or the name of the column in `data` with values to assign marker symbols.
        mode: One of :obj:`~statsplotly.plot_specifiers.data.TraceMode` value.
        axis: One of :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval or the name of the column in `data` with values to specify marker opacity.
        jitter_x: A numeric value to specify jitter amount on the `x` dimension.
        jitter_y: A numeric value to specify jitter amount on the `y` dimension.
        jitter_z: A numeric value to specify jitter amount on the `z` dimension.
        normalizer_x: The normalizer for the `x` dimension. One of :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        normalizer_y: The normalizer for the `y` dimension. One of :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        normalizer_z: The normalizer for the `z` dimension. One of :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        shaded_error: The name of the column in `data` with values to plot continuous error shade.
        error_x: The name of the column in `data` with values to plot error bar in the `x` dimension.
        error_y: The name of the column in `data` with values to plot error bar in the `y` dimension.
        error_z: The name of the column in `data` with values to plot error bar in the `z` dimension.
        fit: One of :obj:`~statsplotly.plot_specifiers.data.RegressionType` value. Computes and plot the corresponding regression.
        size: A numeric value or the name of the column in `data` with values to assign mark sizes.
        x_label: A string to label the x_axis in place of the corresponding column name in `data`.
        y_label: A string to label the y_axis in place of the corresponding column name in `data`.
        z_label: A string to label the y_axis in place of the corresponding column name in `data`.
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

    if (color is not None or size is not None) and mode is None:
        mode = TraceMode.MARKERS
    if color is not None and mode is TraceMode.LINES:
        raise ValueError("Only markers can be mapped to colormap")
    if size is not None and mode is TraceMode.LINES:
        raise ValueError("Size specification only applies to markers")
    if z is not None:
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

    scatter_specifier = ScatterSpecifier(mode=mode, regression_type=fit)

    if opacity is None and scatter_specifier.regression_type is not None:
        logger.debug(
            f"Regression plot is on, setting opacity to {constants.DEFAULT_TRANSPARENCE_OPACITY}"
        )
        opacity = constants.DEFAULT_TRANSPARENCE_OPACITY

    legend_specifier = LegendSpecifier(
        data_pointer=data_handler.data_pointer,
        title=title,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
    )

    color_specifier = ColorSpecifier(
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

    traces: dict[str, BaseTrace] = {}
    traces_data: list[TraceData] = []
    for (slice_name, slice_data), trace_color in zip(
        data_handler.slices_data(),
        color_specifier.get_color_hues(n_colors=data_handler.n_slices),
        strict=True,
    ):
        logger.debug(f"Building {slice_name} trace data...")
        trace_data = TraceData.build_trace_data(
            data=slice_data,
            pointer=data_handler.data_pointer,
            processor=data_processor,
        )

        if data_handler.data_pointer.z is not None:
            traces[slice_name] = go.Scatter3d(
                Scatter3DTrace.build_trace(
                    trace_data=trace_data,
                    trace_name=slice_name,
                    trace_color=trace_color,
                    color_specifier=color_specifier,
                    mode=scatter_specifier.mode,
                ).model_dump()
            )
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
        traces["id_line"] = go.Scatter(
            ScatterTrace.build_id_line(
                x_values=data_handler.get_data("x"),
                y_values=data_handler.get_data("y"),
            ).model_dump()
        )

    coloraxis = color_specifier.build_coloraxis(
        color_data=data_handler.get_data("color"), shared=shared_coloraxis
    )

    layout: "SceneLayout" | "ScatterLayout"
    if data_handler.data_pointer.z is not None:
        layout = SceneLayout.build_layout(axes_specifier=axes_specifier, coloraxis=coloraxis)
    else:
        layout = ScatterLayout.build_layout(axes_specifier=axes_specifier, coloraxis=coloraxis)

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


# Bar plot
def barplot(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
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
    row: int = 1,
    col: int = 1,
) -> go.Figure:
    """Draws a barplot across levels of categorical variable.

    Args:
        data: A :obj:`pandas.DataFrame`
        x: The name of the `x` dimension column in `data`.
        y: The name of the `y` dimension column in `data`.
        slicer: The name of the column in `data` with values to slice the data : one trace is drawn for each level of the `slicer` dimension.
        slice_order: A list of identifiers to order and/or subset data slices specified by `slicer`.
        color: The name of the column in `data` with values to map onto the colormap.
        color_palette:
            - A string refering to a built-in `plotly`, `seaborn` or `matplotlib` colormap.
            - A list of CSS color names or HTML color codes.
            The color palette is used, by order of precedence :
                - To map color data of the `color` parameter onto the corresponding colormap.
                - To assign discrete colors to `slices` of data.
        shared_coloraxis: If True, colorscale is shared across slices of data.
        color_limits: A tuple specifying the (min, max) values of the colormap.
        logscale: A float specifying the log base to use for colorscaling.
        colorbar: If True, draws a colorbar.
        text: A string or the name of the column in `data` with values to appear in the hover tooltip. Column names can be concatenated with '+' to display values from multiple columns. Ignored when `aggregation_func` is not None.
        axis: One of :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval to specify bar opacity.
        barmode: One of :obj:`~statsplotly.plot_specifiers.layout.BarMode` value.
        error_bar: The name of the column in `data` with values to plot error bar.
        aggregation_func: One of :obj:`~statsplotly.plot_specifiers.data.AggregationType` value.
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

    aggregation_specifier = AggregationSpecifier(
        aggregation_func=aggregation_func,
        error_bar=error_bar,
        data_pointer=data_handler.data_pointer,
        data_types=data_handler.data_types,
    )

    color_specifier = ColorSpecifier(
        coloraxis_reference=ColoraxisReference.MAIN_COLORAXIS,
        color_palette=color_palette,
        logscale=logscale,
        colorbar=colorbar,
        color_limits=color_limits,
        opacity=opacity,
    )

    traces: dict[str, BaseTrace] = {}
    traces_data: list[TraceData] = []
    for (slice_name, slice_data), trace_color in zip(
        data_handler.slices_data(),
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

        traces[slice_name] = go.Bar(
            BarTrace.build_trace(
                trace_data=trace_data,
                trace_name=slice_name,
                trace_color=trace_color,
                color_specifier=color_specifier,
            ).model_dump()
        )

        traces_data.append(trace_data)

    legend_specifier = LegendSpecifier(
        data_pointer=data_handler.data_pointer,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_transformation=aggregation_func,
        error_bar=error_bar,
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


# Strip/Box/Violin plot
def catplot(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    slicer: str | None = None,
    slice_order: list[str] | None = None,
    color: str | None = None,
    color_palette: list[str] | str | None = None,
    text: str | None = None,
    marker: str | None = None,
    axis: str | None = None,
    opacity: str | float | None = None,
    plot_type: str = CategoricalPlotType.STRIP,
    jitter_x: float = 1,
    jitter_y: float = 0,
    normalizer: str | None = None,
    size: float = constants.DEFAULT_MARKER_SIZE,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    x_range: Sequence[float | str] | None = None,
    y_range: Sequence[float | str] | None = None,
    fig: go.Figure = None,
    row: int = 1,
    col: int = 1,
) -> go.Figure:
    """Draws a stripplot/boxplot/violinplot across levels of a categorical variable.

    Args:
        data: A :obj:`pandas.DataFrame`
        x: The name of the `x` dimension column in `data`.
        y: The name of the `y` dimension column in `data`.
        slicer: The name of the column in `data` with values to slice the data : one trace is drawn for each level of the `slicer` dimension.
        slice_order: A list of identifiers to order and/or subset data slices specified by `slicer`.
        color: The name of the column in `data` with values to map onto the colormap.
        color_palette:
            - A string refering to a built-in `plotly`, `seaborn` or `matplotlib` colormap.
            - A list of CSS color names or HTML color codes.
            The color palette is used, by order of precedence :
                - To map color data of the `color` parameter onto the corresponding colormap.
                - To assign discrete colors to `slices` of data.
        text: A string or the name of the column in `data` with values to appear in the hover tooltip. Column names can be concatenated with '+' to display values from multiple columns.
        marker: A valid marker symbol or the name of the column in `data` with values to assign marker symbols.
        axis: One of :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval or the name of the column in `data` with values to specify marker opacity.
        plot_type: One of :obj:`~statsplotly.plot_specifiers.trace.CategoricalPlotType` value.
        jitter_x: A numeric value to specify jitter amount on the `x` dimension.
        jitter_y: A numeric value to specify jitter amount on the `y` dimension.
        normalizer: One of :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        size: A numeric value or the name of the column in `data` with values to assign mark sizes.
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

    plot_type = CategoricalPlotType(plot_type)

    data_handler = DataHandler.build_handler(
        data=data,
        data_pointer=DataPointer(
            x=x,
            y=y,
            slicer=slicer,
            color=color,
            text=text,
            marker=marker,
            size=size,
            opacity=opacity,
        ),
        slice_order=slice_order,
    )

    color_specifier = ColorSpecifier(color_palette=color_palette, opacity=opacity)

    if plot_type is CategoricalPlotType.STRIP and is_object_dtype(data_handler.data_types.x):
        x_values_mapping = StripTrace.get_x_strip_map(x_values=data_handler.get_data("x"))
    else:
        x_values_mapping = None
    if jitter_y > 0 and plot_type is not CategoricalPlotType.STRIP:
        logger.warning(f"Jitter parameters have no effect for {plot_type.value}")

    data_processor = DataProcessor(
        x_values_mapping=x_values_mapping,
        jitter_settings=(
            {DataDimension.X: jitter_x, DataDimension.Y: jitter_y}
            if plot_type is CategoricalPlotType.STRIP
            else None
        ),
        normalizer={DataDimension.Y: normalizer},
    )

    traces: dict[str, BaseTrace] = {}
    traces_data: list[TraceData] = []
    for (slice_name, slice_data), trace_color in zip(
        data_handler.slices_data(),
        color_specifier.get_color_hues(n_colors=data_handler.n_slices),
        strict=True,
    ):
        trace_data = TraceData.build_trace_data(
            data=slice_data,
            pointer=data_handler.data_pointer,
            processor=data_processor,
        )

        if plot_type is CategoricalPlotType.STRIP:
            trace = go.Scatter(
                StripTrace.build_trace(
                    trace_data=trace_data,
                    trace_name=slice_name,
                    trace_color=trace_color,
                    color_specifier=color_specifier,
                ).model_dump()
            )

        elif plot_type is CategoricalPlotType.VIOLIN:
            trace = go.Violin(
                ViolinTrace.build_trace(
                    trace_data=trace_data,
                    trace_name=slice_name,
                    trace_color=trace_color,
                    color_specifier=color_specifier,
                ).model_dump()
            )

        elif plot_type is CategoricalPlotType.BOX:
            trace = go.Box(
                BoxTrace.build_trace(
                    trace_data=trace_data,
                    trace_name=slice_name,
                    trace_color=trace_color,
                    color_specifier=color_specifier,
                ).model_dump()
            )
        traces[slice_name] = trace
        traces_data.append(trace_data)

    legend_specifier = LegendSpecifier(
        data_pointer=data_handler.data_pointer,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )

    axes_specifier = AxesSpecifier(
        traces=traces_data,
        axis_format=axis,
        legend=legend_specifier,
        x_range=x_range,
        y_range=y_range,
    )

    layout = CategoricalLayout.build_layout(
        axes_specifier=axes_specifier,
        x_values_map=data_processor.x_values_mapping,
    )

    # Create fig
    fig = create_fig(fig=fig, traces=traces, layout=layout, row=row, col=col)

    return fig


# Histogram plot
def distplot(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    slicer: str | None = None,
    slice_order: list[str] | None = None,
    color_palette: list[str] | str | None = None,
    axis: str | None = None,
    opacity: float = constants.DEFAULT_HISTOGRAM_OPACITY,
    hist: bool = True,
    rug: bool | None = None,
    kde: bool | None = None,
    step: bool | None = None,
    equal_bins: bool | None = None,
    bins: Sequence[float] | int | str = constants.DEFAULT_HISTOGRAM_BIN_COMPUTATION_METHOD,
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
    row: int = 1,
    col: int = 1,
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
        axis: One of :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval to specify bar and line opacity.
        hist: If True, plot histogram bars.
        rug: If True, plot rug bars of the underlying data.
        kde: If True, plot a line of a Kernel Density Estimation of the distribution.
        step: If True, plot a step histogram instead of a standard histogram bars.
        equal_bins: If True, uses the same bins for all `slices` in the data.
        bins: A string or an integer specifying the `bins` parameter for :func:`numpy.histogram`.
        cumulative: If True, draws a cumulative histogram.
        histnorm: One of :obj:`~statsplotly.plot_specifiers.data.HistogramNormType` value.
        central_tendency: One of :obj:`~statsplotly.plot_specifiers.data.CentralTendencyType` value.
        vlines: A dictionary of {slice: (line_name, vertical_coordinates) to draw vertical lines.
        hlines: A dictionary of {slice: (line_name, horizontal_coordinates) to draw horizontal lines.
        barmode: One of :obj:`~statsplotly.plot_specifiers.layout.BarMode` value.
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
        kde=kde,
        step=step,
        bins=bins,
        cumulative=cumulative,
        histnorm=histnorm,
        central_tendency=central_tendency,
        dimension=histogram_dimension,
        data_type=getattr(data_handler.data_types, histogram_dimension),
    )
    if equal_bins:
        # Call histogram on all data to set bin edge attribute
        histogram_specifier.bin_edges = histogram_specifier.histogram_bin_edges(
            data_handler.get_data(histogram_dimension)
        )[0]

    color_specifier = ColorSpecifier(color_palette=color_palette, opacity=opacity)

    traces: dict[str, BaseTrace] = {}
    traces_data: list[TraceData] = []
    for (slice_name, slice_data), trace_color in zip(
        data_handler.slices_data(),
        color_specifier.get_color_hues(n_colors=data_handler.n_slices),
        strict=True,
    ):
        trace_data = TraceData.build_trace_data(data=slice_data, pointer=data_handler.data_pointer)

        traces.update(
            plot_marginal_traces(
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
                traces[line_trace.name] = go.Scatter(line_trace.model_dump())

        if hlines is not None:
            if (hline := hlines.get(slice_name)) is not None:
                line_trace = HistogramLineTrace.build_trace(
                    hline_coordinates=hline,
                    trace_data=trace_data,
                    trace_name=slice_name,
                    trace_color=trace_color,
                    histogram_specifier=histogram_specifier,
                )
                traces[line_trace.name] = go.Scatter(line_trace.model_dump())

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

    if histogram_specifier.central_tendency is not None:
        subplot_col = col + 1 if histogram_dimension is DataDimension.Y else col
        if histogram_specifier.central_tendency is CentralTendencyType.MEAN:
            central_tendency_data = data_handler.get_mean(histogram_dimension)
        elif histogram_specifier.central_tendency is CentralTendencyType.MEDIAN:
            central_tendency_data = data_handler.get_median(histogram_dimension)
        else:
            raise ValueError(
                "Unsupported parameter for distribution central tendency:"
                f" {histogram_specifier.central_tendency.value}"
            )

        if histogram_dimension is DataDimension.X:
            x = histogram_specifier.central_tendency
            y = slicer or "index"
            if fig is None:
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    row_heights=[0.2, 0.8],
                    vertical_spacing=0.05,
                    shared_xaxes=True,
                )
        else:
            x = slicer or "index"
            y = histogram_specifier.central_tendency
            if fig is None:
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    column_widths=[0.8, 0.2],
                    vertical_spacing=0.05,
                    shared_yaxes=True,
                )
        fig = plot(
            fig=fig,
            row=row,
            col=subplot_col,
            data=central_tendency_data,
            x=x,
            y=y,
            slicer=slicer,
            mode=TraceMode.MARKERS,
            color_palette=color_palette,
            error_x=(
                AGG_TO_ERROR_MAPPING[histogram_specifier.central_tendency]
                if histogram_dimension is DataDimension.X
                else None
            ),
            error_y=(
                None
                if histogram_dimension is DataDimension.X
                else AGG_TO_ERROR_MAPPING[histogram_specifier.central_tendency]
            ),
        )
        # Update name
        fig.for_each_trace(
            lambda trace: trace.update(
                name=(
                    f"{trace.name} {histogram_specifier.central_tendency.value} +/-"
                    f" {AGG_TO_ERROR_MAPPING[histogram_specifier.central_tendency].value}"
                )
            )
        )
        # Hide axes
        axis_idx = str(row * subplot_col) if row * subplot_col > 1 else ""
        fig.update_layout({
            f"xaxis{axis_idx}": {"visible": False},
            f"yaxis{axis_idx}": {"visible": False},
        })

    # Create fig
    fig = create_fig(
        fig=fig,
        traces=traces,
        layout=layout,
        row=(
            row + 1
            if histogram_specifier.central_tendency is not None
            and histogram_dimension is DataDimension.X
            else row
        ),
        col=col,
    )

    return fig


# Jointplot
def jointplot(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    slicer: str | None = None,
    slice_order: list[str] | None = None,
    color_palette: list[str] | str | None = None,
    shared_coloraxis: bool = False,
    color_limits: Sequence[float] | None = None,
    logscale: float | None = None,
    colorbar: bool = False,
    text: str | None = None,
    marker: str | None = None,
    mode: str | None = TraceMode.MARKERS,
    axis: str | None = None,
    marginal_plot: str | None = MarginalPlotDimension.ALL,
    kde_color_palette: list[str] | str = constants.DEFAULT_KDE_COLOR_PALETTE,
    hist: bool = True,
    rug: bool | None = None,
    kde: bool | None = None,
    step: bool | None = None,
    equal_bins_x: bool | None = None,
    equal_bins_y: bool | None = None,
    bins_x: Sequence[float] | int | str = constants.DEFAULT_HISTOGRAM_BIN_COMPUTATION_METHOD,
    bins_y: Sequence[float] | int | str = constants.DEFAULT_HISTOGRAM_BIN_COMPUTATION_METHOD,
    histnorm: str | None = None,
    central_tendency: str | None = None,
    barmode: str | None = None,
    plot_type: str = JointplotType.SCATTER,
    opacity: float = constants.DEFAULT_HISTOGRAM_OPACITY,
    jitter_x: float = 0,
    jitter_y: float = 0,
    normalizer_x: str | None = None,
    normalizer_y: str | None = None,
    shaded_error: str | None = None,
    error_x: str | None = None,
    error_y: str | None = None,
    fit: str | None = None,
    size: float | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    x_range: Sequence[float | str] | None = None,
    y_range: Sequence[float | str] | None = None,
    fig: go.Figure = None,
    row: int = 1,
    col: int = 1,
) -> go.Figure:
    """Draws a plot of two variables with bivariate and univariate graphs.

    Args:
        data: A :obj:`pandas.DataFrame`
        x: The name of the `x` dimension column in `data`.
        y: The name of the `y` dimension column in `data`.
        slicer: The name of the column in `data` with values to slice the data : one trace is drawn for each level of the `slicer` dimension.
        slice_order: A list of identifiers to order and/or subset data slices specified by `slicer`.
        color_palette:
            - A string refering to a built-in `plotly`, `seaborn` or `matplotlib` colormap.
            - A list of CSS color names or HTML color codes.
            The color palette is used to assign discrete colors to `slices` of data.
        shared_coloraxis: If True, colorscale is shared across slices of data.
        color_limits: A tuple specifying the (min, max) values of the colormap.
        logscale: A float specifying the log base to use for colorscaling.
        colorbar: If True, draws a colorbar.
        text: A string or the name of the column in `data` with values to appear in the hover tooltip. Column names can be concatenated with '+' to display values from multiple columns.
        marker: A valid marker symbol or the name of the column in `data` with values to assign marker symbols.
        mode: One of :obj:`~statsplotly.plot_specifiers.data.TraceMode` value.
        axis: One of :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval to specify bar and line opacity.
        marginal_plot: One of :obj:`~statsplotly.plot_specifiers.trace.MarginalPlotDimension` value.
        kde_color_palette: The color_palette for the Kernel Density Estimation map.
        hist: If True, plot histogram bars.
        rug: If True, plot rug bars of the underlying data.
        kde: If True, plot a line of a Kernel Density Estimation of the distribution.
        step: If True, plot a step histogram instead of a standard histogram bars.
        equal_bins_x: If True, uses the same bins for `x` dimension of all `slices` in the data.
        equal_bins_y: If True, uses the same bins for `y` dimension of all `slices` in the data.
        bins_x: A string or an integer specifying the `bins` parameter for `x` dimension for :func:`numpy.histogram`.
        bins_y: A string or an integer specifying the `bins` parameter for `y` dimension  for :func:`numpy.histogram`.
        histnorm: One of :obj:`~statsplotly.plot_specifiers.data.HistogramNormType` value.
        central_tendency: One of :obj:`~statsplotly.plot_specifiers.data.CentralTendencyType` value.
        barmode: One of :obj:`~statsplotly.plot_specifiers.layout.BarMode` value.
        plot_type: One of :obj:`~statsplotly.plot_specifiers.trace.JointplotType` value.
        opacity: A numeric value in the (0, 1) interval to specify marker opacity.
        jitter_x: A numeric value to specify jitter amount on the `x` dimension.
        jitter_y: A numeric value to specify jitter amount on the `y` dimension.
        normalizer_x: The normalizer for the `x` dimension. One of :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        normalizer_y: The normalizer for the `y` dimension. One of :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        shaded_error: The name of the column in `data` with values to plot continuous error shade.
        error_x: The name of the column in `data` with values to plot error bar in the `x` dimension.
        error_y: The name of the column in `data` with values to plot error bar in the `y` dimension.
        fit: One of :obj:`~statsplotly.plot_specifiers.data.RegressionType` value. Computes and plot the corresponding regression.
        size: A numeric value or the name of the column in `data` with values to assign mark sizes.
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
        data_pointer=DataPointer(
            x=x,
            y=y,
            slicer=slicer,
            shaded_error=shaded_error,
            error_x=error_x,
            error_y=error_y,
            text=text,
            marker=marker,
            size=size,
            opacity=opacity,
        ),
        slice_order=slice_order,
    )
    if fit is not None and not (
        is_numeric_dtype(data_handler.data_types.x) and is_numeric_dtype(data_handler.data_types.y)
    ):
        raise StatsPlotSpecificationError(f"{fit} regression requires numeric dtypes")

    jointplot_specifier = JointplotSpecifier(
        plot_type=plot_type,
        marginal_plot=marginal_plot,
        scatter_specifier=ScatterSpecifier(mode=mode, regression_type=fit),
    )

    def specify_marginal_histogram(
        dimension: DataDimension,
        bins: Sequence[float] | int | str,
        equal_bins: bool | None,
    ) -> HistogramSpecifier:
        histogram_specifier = HistogramSpecifier(
            hist=hist,
            rug=rug,
            kde=kde,
            step=step,
            bins=bins,
            central_tendency=central_tendency,
            histnorm=histnorm,
            data_type=getattr(data_handler.data_types, dimension),
            dimension=dimension,
        )
        if equal_bins:
            histogram_specifier.bin_edges = histogram_specifier.histogram_bin_edges(
                data_handler.get_data(dimension)
            )[0]
        return histogram_specifier

    histogram_specifiers: dict[DataDimension, HistogramSpecifier] = {}
    if jointplot_specifier.marginal_plot in (
        MarginalPlotDimension.ALL,
        MarginalPlotDimension.X,
    ) or jointplot_specifier.plot_type in (
        JointplotType.SCATTER_KDE,
        JointplotType.KDE,
        JointplotType.SCATTER_KDE,
        JointplotType.HISTOGRAM,
        JointplotType.Y_HISTMAP,
    ):
        histogram_specifiers[DataDimension.X] = specify_marginal_histogram(
            dimension=DataDimension.X, bins=bins_x, equal_bins=equal_bins_x
        )

    if jointplot_specifier.marginal_plot in (
        MarginalPlotDimension.ALL,
        MarginalPlotDimension.Y,
    ) or jointplot_specifier.plot_type in (
        JointplotType.SCATTER_KDE,
        JointplotType.KDE,
        JointplotType.HISTOGRAM,
        JointplotType.X_HISTMAP,
    ):
        histogram_specifiers[DataDimension.Y] = specify_marginal_histogram(
            dimension=DataDimension.Y, bins=bins_y, equal_bins=equal_bins_y
        )
    jointplot_specifier.histogram_specifier = histogram_specifiers

    if opacity is None and jointplot_specifier.scatter_specifier.regression_type is not None:
        logger.debug(
            f"Regression plot is on, setting opacity to {constants.DEFAULT_TRANSPARENCE_OPACITY}"
        )
        opacity = constants.DEFAULT_TRANSPARENCE_OPACITY

    sliced_data_color_specifier = ColorSpecifier(
        coloraxis_reference=ColoraxisReference.MAIN_COLORAXIS,
        color_palette=color_palette,
        logscale=logscale,
        colorbar=colorbar,
        color_limits=color_limits,
        opacity=opacity,
    )
    main_data_color_specifier = ColorSpecifier(
        color_palette=kde_color_palette,
        logscale=logscale,
        color_limits=color_limits,
        opacity=opacity,
    )

    data_processor = DataProcessor(
        jitter_settings={DataDimension.X: jitter_x, DataDimension.Y: jitter_y},
        normalizer={
            DataDimension.X: normalizer_x,
            DataDimension.Y: normalizer_y,
        },
    )

    global_main_traces: dict[str, BaseTrace] = {}
    slices_main_traces: dict[str, BaseTrace] = {}
    slices_marginal_traces: dict[str, BaseTrace] = {}
    preplotted_traces: dict[str, BaseTrace] = {}

    traces_data: list[TraceData] = []

    # Global trace
    if data_handler.n_slices > 1:
        global_main_traces.update(
            plot_jointplot_main_traces(
                trace_data=TraceData.build_trace_data(
                    data=data_handler.data,
                    pointer=data_handler.data_pointer,
                    processor=data_processor,
                ),
                trace_name=SliceTraceType.ALL_DATA.value,
                trace_color="grey",
                color_specifier=main_data_color_specifier,
                jointplot_specifier=jointplot_specifier,
            )
        )

    # Slice trace
    for (slice_name, slice_data), trace_color in zip(
        data_handler.slices_data(),
        sliced_data_color_specifier.get_color_hues(n_colors=data_handler.n_slices),
        strict=True,
    ):
        trace_data = TraceData.build_trace_data(
            data=slice_data,
            pointer=data_handler.data_pointer,
            processor=data_processor,
        )

        slices_main_traces.update(
            plot_jointplot_main_traces(
                trace_data=trace_data,
                trace_name=slice_name,
                trace_color=trace_color,
                color_specifier=main_data_color_specifier,
                jointplot_specifier=jointplot_specifier,
            )
        )

        if jointplot_specifier.plot_scatter:
            slices_main_traces.update(
                plot_scatter_traces(
                    trace_data=trace_data,
                    trace_name=slice_name,
                    trace_color=trace_color,
                    color_specifier=sliced_data_color_specifier,
                    scatter_specifier=jointplot_specifier.scatter_specifier,
                )
            )

        # X and Y histograms
        for dimension in [DataDimension.X, DataDimension.Y]:
            if (
                jointplot_specifier.marginal_plot == dimension
                or jointplot_specifier.marginal_plot is MarginalPlotDimension.ALL
            ):
                jointplot_specifier.histogram_specifier[dimension].dimension = dimension
                slices_marginal_traces.update(
                    plot_marginal_traces(
                        trace_data=trace_data,
                        trace_name=slice_name,
                        trace_color=trace_color,
                        color_specifier=sliced_data_color_specifier,
                        histogram_specifier=jointplot_specifier.histogram_specifier[dimension],
                    )
                )

        traces_data.append(trace_data)

    # Adjust legends
    adjust_jointplot_legends(jointplot_specifier, slices_marginal_traces)

    legend_specifier = partial(
        LegendSpecifier,
        data_pointer=data_handler.data_pointer,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )

    axes_specifier = AxesSpecifier(
        traces=traces_data,
        axis_format=axis,
        legend=legend_specifier(),
        x_range=x_range,
        y_range=y_range,
    )
    if axes_specifier.axis_format is AxisFormat.ID_LINE:
        slices_main_traces["id_line"] = go.Scatter(
            ScatterTrace.build_id_line(
                x_values=data_handler.get_data("x"),
                y_values=data_handler.get_data("y"),
            ).model_dump()
        )

    coloraxis = sliced_data_color_specifier.build_coloraxis(
        color_data=data_handler.get_data("color"), shared=shared_coloraxis
    )

    if fig is None:
        fig = make_subplots(
            rows=2 if jointplot_specifier.plot_x_distribution else 1,
            cols=2 if jointplot_specifier.plot_y_distribution else 1,
            row_heights=[0.2, 0.8] if jointplot_specifier.plot_x_distribution else [1],
            column_widths=[0.8, 0.2] if jointplot_specifier.plot_y_distribution else [1],
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            shared_xaxes=True,
            shared_yaxes=True,
        )
    else:
        preplotted_traces.update({trace.name: trace for trace in fig.data})

    main_row = row + 1 if jointplot_specifier.plot_x_distribution else row
    # Plot main trace
    fig = create_fig(
        fig=fig,
        traces={**slices_main_traces, **global_main_traces},
        layout=ScatterLayout.build_layout(axes_specifier=axes_specifier, coloraxis=coloraxis),
        row=main_row,
        col=col,
    )
    if data_handler.n_slices > 0:
        fig.update_traces(showlegend=True)

    def add_marginal_distribution_to_layout(dimension: DataDimension, fig: go.Figure) -> go.Figure:
        marginal_row = main_row if dimension is DataDimension.Y else row
        marginal_col = col + 1 if dimension is DataDimension.Y else col

        axes_specifier = AxesSpecifier(
            traces=traces_data,
            legend=legend_specifier(
                x_transformation=(
                    jointplot_specifier.histogram_specifier[  # type: ignore
                        DataDimension.Y
                    ].histnorm
                    if dimension is DataDimension.Y
                    else None
                ),
                y_transformation=(
                    jointplot_specifier.histogram_specifier[  # type: ignore
                        DataDimension.X
                    ].histnorm
                    if dimension is DataDimension.X
                    else None
                ),
            ),
            x_range=x_range,
            y_range=y_range,
        )

        # Plot histogram traces
        fig = create_fig(
            fig=fig,
            traces={
                name: trace for name, trace in slices_marginal_traces.items() if dimension in name
            },
            layout=HistogramLayout.build_layout(axes_specifier=axes_specifier, barmode=barmode),
            row=marginal_row,
            col=marginal_col,
        )
        # Update layout
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            row=marginal_row,
            col=marginal_col,
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            row=marginal_row,
            col=marginal_col,
        )
        return fig

    # Marginals
    if jointplot_specifier.plot_x_distribution:
        fig = add_marginal_distribution_to_layout(dimension=DataDimension.X, fig=fig)

    if jointplot_specifier.plot_y_distribution:
        fig = add_marginal_distribution_to_layout(dimension=DataDimension.Y, fig=fig)

    # Add menus
    if len(global_main_traces) > 0:
        fig = add_update_menu(
            fig=fig,
            data_handler=data_handler,
            slices_traces=(
                {
                    **slices_marginal_traces,
                    **{
                        trace_name: trace_value
                        for trace_name, trace_value in slices_main_traces.items()
                        if isinstance(trace_value, go.Scatter)
                    },
                }
                if jointplot_specifier.plot_scatter
                else slices_marginal_traces
            ),
            preplotted_traces=preplotted_traces,
        )

    return fig


# Heatmap plot
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
    row: int = 1,
    col: int = 1,
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
                - To map color data of the `color` parameter onto the corresponding colormap.
                - To assign discrete colors to `slices` of data.
        shared_coloraxis: If True, colorscale is shared across slices of data.
        color_limits: A tuple specifying the (min, max) values of the colormap.
        logscale: A float specifying the log base to use for colorscaling.
        colorbar: If True, draws a colorbar.
        text: A string or the name of the column in `data` with values to appear in the hover tooltip. Column names can be concatenated with '+' to display values from multiple columns.
        axis: One of :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval to specify heatmap opacity.
        normalizer: The normalizer for the `z` dimension. One of :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        x_label: A string to label the x_axis in place of the corresponding column name in `data`.
        y_label: A string to label the y_axis in place of the corresponding column name in `data`.
        z_label: A string to label the y_axis in place of the corresponding column name in `data`.
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

    color_specifier = ColorSpecifier(
        coloraxis_reference=ColoraxisReference.MAIN_COLORAXIS,
        color_palette=color_palette,
        logscale=logscale,
        color_limits=color_limits,
        colorbar=colorbar,
        opacity=opacity,
    )

    data_processor = DataProcessor(normalizer={DataDimension.Z: normalizer})

    traces: dict[str, BaseTrace] = {}
    traces_data: list[TraceData] = []

    if data_handler.n_slices > 1:
        global_trace = HeatmapTrace.build_trace(
            trace_data=TraceData.build_trace_data(
                data=data_handler.data,
                pointer=data_handler.data_pointer,
            ),
            trace_name=SliceTraceType.ALL_DATA,
            color_specifier=color_specifier,
        )
        traces[global_trace.name] = go.Heatmap(global_trace.model_dump())

    for slice_name, slice_data in data_handler.slices_data():
        trace_data = TraceData.build_trace_data(
            data=slice_data,
            pointer=data_handler.data_pointer,
            processor=data_processor,
        )

        traces[slice_name] = go.Heatmap(
            HeatmapTrace.build_trace(
                trace_data=trace_data,
                trace_name=slice_name,
                color_specifier=color_specifier,
            ).model_dump()
        )
        traces_data.append(trace_data)

    legend_specifier = LegendSpecifier(
        data_pointer=data_handler.data_pointer,
        title=title,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
    )

    axes_specifier = AxesSpecifier(
        traces=traces_data,
        axis_format=axis,
        legend=legend_specifier,
        x_range=x_range,
        y_range=y_range,
    )

    coloraxis = color_specifier.build_coloraxis(
        color_data=data_handler.get_data("z"), shared=shared_coloraxis
    )

    layout = HeatmapLayout.build_layout(
        axes_specifier=axes_specifier,
        coloraxis=coloraxis,
    )

    # Create fig
    fig = create_fig(fig=fig, traces=traces, layout=layout, row=row, col=col)

    # Add menus
    if len(data_handler.slice_levels) > 0:
        fig = add_update_menu(fig=fig, data_handler=data_handler)

    return fig
