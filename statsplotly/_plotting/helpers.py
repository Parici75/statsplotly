"""Helper modules for plotting routines."""

import plotly

from statsplotly.plot_objects.trace import (
    ContourTrace,
    EcdfTrace,
    HeatmapTrace,
    Histogram2dTrace,
    HistogramTrace,
    KdeTrace,
    RugTrace,
    ScatterTrace,
    StepHistogramTrace,
)
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.plot_specifiers.data import TraceData
from statsplotly.plot_specifiers.layout import set_horizontal_colorbar
from statsplotly.plot_specifiers.trace import (
    HistogramSpecifier,
    JointplotSpecifier,
    JointplotType,
    ScatterSpecifier,
)


def plot_jointplot_main_traces(
    trace_data: TraceData,
    trace_name: str,
    trace_color: str,
    color_specifier: ColorSpecifier,
    jointplot_specifier: JointplotSpecifier,
) -> dict[str, plotly.basedatatypes.BaseTraceType]:
    """Constructs the main traces of a jointplot layout."""

    traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}
    if jointplot_specifier.plot_kde:
        contour_trace = ContourTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            color_specifier=color_specifier,
            jointplot_specifier=jointplot_specifier,
        )
        traces[contour_trace.name] = contour_trace.to_plotly_trace()

    if jointplot_specifier.plot_type is JointplotType.HISTOGRAM:
        histogram_trace = Histogram2dTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            jointplot_specifier=jointplot_specifier,
        )
        # Make colorbars horizontal
        if histogram_trace.colorbar is not None:
            histogram_trace.colorbar = set_horizontal_colorbar(histogram_trace.colorbar)

        traces[histogram_trace.name] = histogram_trace.to_plotly_trace()

    if jointplot_specifier.plot_type in (
        JointplotType.X_HISTMAP,
        JointplotType.Y_HISTMAP,
    ):
        heatmap_trace = HeatmapTrace.build_histmap_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            color_specifier=color_specifier,
            jointplot_specifier=jointplot_specifier,
        )
        # Make colorbars horizontal
        # TODO: Get read of this one when we use _SubplotGridCommonColoraxisFormatter class for managing coloraxis
        if heatmap_trace.colorbar is not None:
            heatmap_trace.colorbar = set_horizontal_colorbar(heatmap_trace.colorbar)

        traces[heatmap_trace.name] = heatmap_trace.to_plotly_trace()

    return traces


def plot_scatter_traces(
    trace_data: TraceData,
    trace_name: str,
    trace_color: str,
    color_specifier: ColorSpecifier,
    scatter_specifier: ScatterSpecifier,
) -> dict[str, plotly.basedatatypes.BaseTraceType]:
    """Constructs scatter traces."""

    traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}
    traces[trace_name] = ScatterTrace.build_trace(
        trace_data=trace_data,
        trace_name=trace_name,
        trace_color=trace_color,
        color_specifier=color_specifier,
        mode=scatter_specifier.mode,
    ).to_plotly_trace()

    if trace_data.shaded_error is not None:
        upper_bound_trace = ScatterTrace.build_upper_error_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
        )
        traces[upper_bound_trace.name] = upper_bound_trace.to_plotly_trace()

        lower_bound_trace = ScatterTrace.build_lower_error_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
        )
        traces[lower_bound_trace.name] = lower_bound_trace.to_plotly_trace()

    if scatter_specifier.regression_type is not None:
        regression_trace = ScatterTrace.build_regression_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            regression_type=scatter_specifier.regression_type,
        )
        traces[regression_trace.name] = regression_trace.to_plotly_trace()

    return traces


def plot_distplot_traces(
    trace_data: TraceData,
    trace_name: str,
    trace_color: str,
    color_specifier: ColorSpecifier,
    histogram_specifier: HistogramSpecifier,
) -> dict[str, plotly.basedatatypes.BaseTraceType]:
    """Constructs distplot traces."""

    traces: dict[str, plotly.basedatatypes.BaseTraceType] = {}
    if histogram_specifier.dimension is None:
        raise ValueError("`histogram_specifier.dimension` can not be `None`")

    if histogram_specifier.hist:
        hist_trace: StepHistogramTrace | HistogramTrace
        if histogram_specifier.step:
            hist_trace = StepHistogramTrace.build_trace(
                trace_data=trace_data,
                trace_name=trace_name,
                trace_color=trace_color,
                color_specifier=color_specifier,
                histogram_specifier=histogram_specifier,
            )

        else:
            hist_trace = HistogramTrace.build_trace(
                trace_data=trace_data,
                trace_name=trace_name,
                trace_color=trace_color,
                color_specifier=color_specifier,
                histogram_specifier=histogram_specifier,
            )
        traces["_".join((hist_trace.name, histogram_specifier.dimension))] = (
            hist_trace.to_plotly_trace()
        )

    if histogram_specifier.ecdf:
        ecdf_trace = EcdfTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            histogram_specifier=histogram_specifier,
        )
        traces["_".join((ecdf_trace.name, histogram_specifier.dimension))] = (
            ecdf_trace.to_plotly_trace()
        )

    if histogram_specifier.rug:
        rug_trace = RugTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            histogram_specifier=histogram_specifier,
        )
        traces["_".join((rug_trace.name, histogram_specifier.dimension))] = (
            rug_trace.to_plotly_trace()
        )

    if histogram_specifier.kde:
        kde_trace = KdeTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            histogram_specifier=histogram_specifier,
        )
        traces["_".join((kde_trace.name, histogram_specifier.dimension))] = (
            kde_trace.to_plotly_trace()
        )

    # Make sure we show at least one legend item
    if not any(
        trace.showlegend if trace.showlegend is not None else True for trace in traces.values()
    ):
        list(traces.values())[-1].update(showlegend=True)

    return traces
