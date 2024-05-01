"""Strip/Box/Violin plots"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio

from statsplotly import constants
from statsplotly.plot_objects.layout import CategoricalLayout
from statsplotly.plot_objects.trace import BoxTrace, StripTrace, ViolinTrace

# Specifiers
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.plot_specifiers.data import (
    DataHandler,
    DataPointer,
    DataProcessor,
    TraceData,
)
from statsplotly.plot_specifiers.figure import create_fig
from statsplotly.plot_specifiers.layout import (
    AxesSpecifier,
    ColoraxisReference,
    LegendSpecifier,
)

# Trace objects
from statsplotly.plot_specifiers.trace import (
    CategoricalPlotSpecifier,
    CategoricalPlotType,
)

pio.templates.default = constants.DEFAULT_TEMPLATE
np.seterr(invalid="ignore")

logger = logging.getLogger(__name__)


def catplot(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    orientation: str | None = None,
    slicer: str | None = None,
    slice_order: list[str] | None = None,
    color: str | None = None,
    color_palette: list[str] | str | None = None,
    shared_coloraxis: bool = False,
    text: str | None = None,
    marker: str | None = None,
    axis: str | None = None,
    opacity: str | float | None = None,
    plot_type: str | None = None,
    jitter: float | None = None,
    normalizer: str | None = None,
    size: float = constants.DEFAULT_MARKER_SIZE,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    x_range: Sequence[float | str] | None = None,
    y_range: Sequence[float | str] | None = None,
    fig: go.Figure = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Draws a stripplot/boxplot/violinplot across levels of a categorical variable.

    Args:
        data: A :obj:`pandas.DataFrame`
        x: The name of the `x` dimension column in `data`.
        y: The name of the `y` dimension column in `data`.
        orientation: A :obj:`~Astatsplotly.plot_specifiers.trace.PlotOrientation` value to force the orientation of the plot.
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
        text: A string or the name of the column in `data` with values to appear in the hover tooltip. Column names can be concatenated with '+' to display values from multiple columns.
        marker: A valid marker symbol or the name of the column in `data` with values to assign marker symbols.
        axis: A :obj:`~statsplotly.plot_specifiers.layout.AxisFormat` value.
        opacity: A numeric value in the (0, 1) interval or the name of the column in `data` with values to specify marker opacity.
        plot_type: A :obj:`~statsplotly.plot_specifiers.trace.CategoricalPlotType` value.
        jitter: A numeric value to specify jitter amount on the categorical dimension.
        normalizer: A :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value to normalize data on the continous dimension.
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
            color=color,
            text=text,
            marker=marker,
            size=size,
            opacity=opacity,
        ),
        slice_order=slice_order,
    )

    categorical_plot_specifier = CategoricalPlotSpecifier(
        plot_type=plot_type, prefered_orientation=orientation, data_types=data_handler.data_types
    )

    if jitter is not None and categorical_plot_specifier.plot_type is not CategoricalPlotType.STRIP:
        logger.warning(
            f"Jitter parameters have no effect for {categorical_plot_specifier.plot_type.value}"
        )

    color_specifier = ColorSpecifier.build_from_color_data(
        color_data=data_handler.get_data("color"),
        coloraxis_reference=ColoraxisReference.MAIN_COLORAXIS,
        color_palette=color_palette,
        opacity=opacity,
    )

    data_processor = DataProcessor(
        data_values_map=categorical_plot_specifier.get_category_strip_map(data_handler),
        jitter_settings=(
            {
                categorical_plot_specifier.anchor_dimension: jitter
                or constants.DEFAULT_STRIPPLOT_JITTER
            }
            if categorical_plot_specifier.plot_type is CategoricalPlotType.STRIP
            else None
        ),
        normalizer={categorical_plot_specifier.anchored_dimension: normalizer},
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

        trace_constructor: Any
        match categorical_plot_specifier.plot_type:
            case CategoricalPlotType.STRIP:
                trace_constructor = StripTrace

            case CategoricalPlotType.VIOLIN:
                trace_constructor = ViolinTrace

            case CategoricalPlotType.BOX:
                trace_constructor = BoxTrace

        traces[slice_name] = trace_constructor.build_trace(
            trace_data=trace_data,
            trace_name=slice_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            categorical_plot_specifier=categorical_plot_specifier,
        ).to_plotly_trace()

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

    coloraxis = color_specifier.build_coloraxis(
        color_data=data_handler.get_data("color"), shared=shared_coloraxis
    )

    layout = CategoricalLayout.build_layout(
        axes_specifier=axes_specifier,
        categorical_values_map=data_processor.data_values_map,
        coloraxis=coloraxis,
    )

    # Create fig
    fig = create_fig(fig=fig, traces=traces, layout=layout, row=row, col=col)

    return fig
