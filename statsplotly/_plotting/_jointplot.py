"""Jointplots"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio
from pandas.api.types import is_numeric_dtype
from pydantic import ValidationError

from statsplotly import constants
from statsplotly.exceptions import StatsPlotSpecificationError
from statsplotly.plot_objects.layout import HistogramLayout, ScatterLayout
from statsplotly.plot_objects.trace import ScatterTrace

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
from statsplotly.plot_specifiers.figure import JointplotPlot, create_fig
from statsplotly.plot_specifiers.layout import (
    AxesSpecifier,
    AxisFormat,
    ColoraxisReference,
    LegendSpecifier,
    add_update_menu,
    adjust_jointplot_legends,
)

# Trace objects
from statsplotly.plot_specifiers.trace import (
    HistogramSpecifier,
    JointplotSpecifier,
    JointplotType,
    MarginalPlotDimension,
    ScatterSpecifier,
    TraceMode,
)

# Helpers
from .helpers import (
    plot_distplot_traces,
    plot_jointplot_main_traces,
    plot_scatter_traces,
)

pio.templates.default = constants.DEFAULT_TEMPLATE
np.seterr(invalid="ignore")

logger = logging.getLogger(__name__)


def jointplot(
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
    marker: str | None = None,
    mode: str | None = TraceMode.MARKERS,
    axis: str | None = None,
    marginal_plot: str | None = MarginalPlotDimension.ALL,
    kde_color_palette: list[str] | str = constants.DEFAULT_KDE_COLOR_PALETTE,
    hist: bool = True,
    rug: bool | None = None,
    ecdf: bool | None = None,
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
    size: float | str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    x_range: Sequence[float | str] | None = None,
    y_range: Sequence[float | str] | None = None,
    fig: go.Figure = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Draws a plot of two variables with bivariate and univariate graphs.

    Args:
        data: A :obj:`pandas.DataFrame`
        x: The name of the `x` dimension column in `data`.
        y: The name of the `y` dimension column in `data`.
        slicer: The name of the column in `data` with values to slice the data : one trace is drawn for each level of the `slicer` dimension.
        slice_order: A list of identifiers to order and/or subset data slices specified by `slicer`.
        color: The name of the column in `data` with values to map onto the colormap. Specifying a `color` along with `marginal != None` raises a `StatsPlotSpecificationError`.
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
        opacity: A numeric value in the (0, 1) interval to specify bar and line opacity.
        marginal_plot: A :obj:`~statsplotly.plot_specifiers.trace.MarginalPlotDimension` value.
        kde_color_palette: The color_palette for the Kernel Density Estimation map.
        hist: If True, plot histogram bars.
        rug: If True, plot rug bars of the underlying data.
        ecdf: If True, plot the Empirical Cumulative Density Function.
        kde: If True, plot a line of a Kernel Density Estimation of the distribution.
        step: If True, plot a step histogram instead of a standard histogram bars.
        equal_bins_x: If True, uses the same bins for the `x` dimension of all `slices` in the data.
        equal_bins_y: If True, uses the same bins for the `y` dimension of all `slices` in the data.
        bins_x: A string, integer, or sequence specifying the `bins` parameter for the `x` dimension for :func:`numpy.histogram`.
        bins_y: A string, integer, or sequence specifying the `bins` parameter for the `y` dimension  for :func:`numpy.histogram`.
        histnorm: A :obj:`~statsplotly.plot_specifiers.data.HistogramNormType` value.
        central_tendency: A :obj:`~statsplotly.plot_specifiers.data.CentralTendencyType` value.
        barmode: A :obj:`~statsplotly.plot_specifiers.layout.BarMode` value.
        plot_type: A :obj:`~statsplotly.plot_specifiers.trace.JointplotType` value.
        opacity: A numeric value in the (0, 1) interval to specify marker opacity.
        jitter_x: A numeric value to specify jitter amount on the `x` dimension.
        jitter_y: A numeric value to specify jitter amount on the `y` dimension.
        normalizer_x: The normalizer for the `x` dimension. A :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        normalizer_y: The normalizer for the `y` dimension. A :obj:`~statsplotly.plot_specifiers.data.NormalizationType` value.
        shaded_error: The name of the column in `data` with values to plot continuous error shade.
        error_x: The name of the column in `data` with values to plot error bar in the `x` dimension.
        error_y: The name of the column in `data` with values to plot error bar in the `y` dimension.
        fit: A :obj:`~statsplotly.plot_specifiers.data.RegressionType` value. Computes and plot the corresponding regression.
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
        scatter_specifier=ScatterSpecifier(
            mode=mode, regression_type=fit, data_types=data_handler.data_types
        ),
    )
    if data_handler.data_pointer.color is not None:
        if jointplot_specifier.plot_type in (JointplotType.HISTOGRAM, JointplotType.KDE):
            logger.warning(
                f"Color mapping have no effect with `plot_type={jointplot_specifier.plot_type.value}`"
            )

    def specify_marginal_histogram(
        dimension: DataDimension,
        bins: Sequence[float] | int | str,
        equal_bins: bool | None,
    ) -> HistogramSpecifier:
        try:
            histogram_specifier = HistogramSpecifier(
                hist=hist,
                rug=rug,
                ecdf=ecdf,
                kde=kde,
                step=step,
                bins=bins,
                central_tendency=central_tendency,
                histnorm=histnorm,
                data_type=getattr(data_handler.data_types, dimension),
                dimension=dimension,
            )
        except ValidationError as exc:
            raise StatsPlotSpecificationError(
                f"Error when initializing marginal histogram for {dimension.value} dimension, try to change `marginal_plot` argument"
            ) from exc

        if equal_bins:
            histogram_specifier.bin_edges = histogram_specifier.get_histogram_bin_edges(
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

    sliced_data_color_specifier = ColorSpecifier.build_from_color_data(
        color_data=data_handler.get_data("color"),
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

    global_main_traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}
    slices_main_traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}
    slices_marginal_traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}
    preplotted_traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}

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
        data_handler.iter_slices(),
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
                jointplot_specifier.marginal_plot == dimension  # type: ignore[comparison-overlap]
                or jointplot_specifier.marginal_plot is MarginalPlotDimension.ALL
            ):
                jointplot_specifier.histogram_specifier[dimension].dimension = dimension
                slices_marginal_traces.update(
                    plot_distplot_traces(
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

    main_legend_specifier = LegendSpecifier(
        data_pointer=data_handler.data_pointer,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )

    axes_specifier = AxesSpecifier(
        traces=traces_data,
        axis_format=axis,
        legend=main_legend_specifier,
        x_range=x_range,
        y_range=y_range,
    )
    if axes_specifier.axis_format is AxisFormat.ID_LINE:
        slices_main_traces["id_line"] = ScatterTrace.build_id_line(
            x_values=data_handler.get_data("x"),
            y_values=data_handler.get_data("y"),
        ).to_plotly_trace()

    coloraxis = sliced_data_color_specifier.build_coloraxis(
        color_data=data_handler.get_data("color"), shared=shared_coloraxis
    )

    figure_plot = JointplotPlot.initialize(
        plot_specifier=jointplot_specifier, fig=fig, row=row, col=col
    )
    preplotted_traces.update({trace.name: trace for trace in figure_plot.fig.data})

    # Plot main trace
    fig = create_fig(
        fig=figure_plot.fig,
        traces={**slices_main_traces, **global_main_traces},
        layout=ScatterLayout.build_layout(axes_specifier=axes_specifier, coloraxis=coloraxis),
        row=figure_plot.main_row,
        col=figure_plot.col,
    )
    if data_handler.n_slices > 0:
        fig.update_traces(showlegend=True)

    def add_marginal_distribution_to_layout(
        dimension: DataDimension, figure_plot: JointplotPlot
    ) -> None:
        marginal_row = figure_plot.main_row if dimension is DataDimension.Y else figure_plot.row
        marginal_col = figure_plot.col + 1 if dimension is DataDimension.Y else figure_plot.col

        _data_pointer = data_handler.data_pointer.copy()
        if dimension is DataDimension.Y:
            _data_pointer.x = None
            _data_pointer.y = y
            _legend_specifier = LegendSpecifier(
                data_pointer=_data_pointer,
                x_transformation=jointplot_specifier.histogram_specifier[
                    DataDimension.Y
                ].histnorm,  # type: ignore
            )

        elif dimension is DataDimension.X:
            _data_pointer.x = x
            _data_pointer.y = None
            _legend_specifier = LegendSpecifier(
                data_pointer=_data_pointer,
                y_transformation=jointplot_specifier.histogram_specifier[
                    DataDimension.X
                ].histnorm,  # type: ignore
            )

        axes_specifier = AxesSpecifier(
            traces=traces_data,
            legend=_legend_specifier,
            x_range=x_range,
            y_range=y_range,
        )

        # Plot histogram traces
        fig = create_fig(
            fig=figure_plot.fig,
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

    # Marginals
    if jointplot_specifier.plot_x_distribution:
        add_marginal_distribution_to_layout(dimension=DataDimension.X, figure_plot=figure_plot)

    if jointplot_specifier.plot_y_distribution:
        add_marginal_distribution_to_layout(dimension=DataDimension.Y, figure_plot=figure_plot)

    # Finalize figure
    figure_plot.tidy_plot()

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
