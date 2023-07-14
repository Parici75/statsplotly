"""Helper modules for plotting routines."""

from plotly import graph_objs as go

from statsplotly.plot_objects.trace_objects import (
    BaseTrace,
    ContourTrace,
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
) -> dict[str, BaseTrace]:
    """Constructs the main traces of a jointplot layout."""

    traces: dict[str, BaseTrace] = {}
    if jointplot_specifier.plot_kde:
        contour_trace = ContourTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            color_specifier=color_specifier,
            jointplot_specifier=jointplot_specifier,
        )
        traces[contour_trace.name] = go.Contour(contour_trace.model_dump())

    if jointplot_specifier.plot_type is JointplotType.HISTOGRAM:
        histogram_trace = Histogram2dTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            jointplot_specifier=jointplot_specifier,
        )
        traces[histogram_trace.name] = go.Histogram2d(histogram_trace.model_dump())

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
        traces[heatmap_trace.name] = go.Heatmap(heatmap_trace.model_dump())

    return traces


def plot_scatter_traces(
    trace_data: TraceData,
    trace_name: str,
    trace_color: str,
    color_specifier: ColorSpecifier,
    scatter_specifier: ScatterSpecifier,
) -> dict[str, BaseTrace]:
    """Constructs scatter traces."""

    traces: dict[str, BaseTrace] = {}
    traces[trace_name] = go.Scatter(
        **ScatterTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            mode=scatter_specifier.mode,
        ).model_dump()
    )

    if trace_data.shaded_error is not None:
        upper_bound_trace = ScatterTrace.build_upper_error_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
        )
        traces[upper_bound_trace.name] = go.Scatter(**upper_bound_trace.model_dump())

        lower_bound_trace = ScatterTrace.build_lower_error_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
        )
        traces[lower_bound_trace.name] = go.Scatter(**lower_bound_trace.model_dump())

    if scatter_specifier.regression_type is not None:
        regression_trace = ScatterTrace.build_regression_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            regression_type=scatter_specifier.regression_type,
        )
        traces[regression_trace.name] = go.Scatter(**regression_trace.model_dump())

    return traces


def plot_marginal_traces(
    trace_data: TraceData,
    trace_name: str,
    trace_color: str,
    color_specifier: ColorSpecifier,
    histogram_specifier: HistogramSpecifier,
) -> dict[str, BaseTrace]:
    """Constructs traces of marginal distributions."""

    traces: dict[str, BaseTrace] = {}
    if histogram_specifier.dimension is None:
        raise ValueError("`histogram_specifier.dimension` can not be `None`")

    if histogram_specifier.hist:
        if histogram_specifier.step:
            traces["_".join((trace_name, histogram_specifier.dimension))] = go.Scatter(
                StepHistogramTrace.build_trace(
                    trace_data=trace_data,
                    trace_name=trace_name,
                    trace_color=trace_color,
                    color_specifier=color_specifier,
                    histogram_specifier=histogram_specifier,
                ).model_dump()
            )
        else:
            traces["_".join((trace_name, histogram_specifier.dimension))] = go.Histogram(
                HistogramTrace.build_trace(
                    trace_data=trace_data,
                    trace_name=trace_name,
                    trace_color=trace_color,
                    color_specifier=color_specifier,
                    histogram_specifier=histogram_specifier,
                ).model_dump()
            )

    if histogram_specifier.rug:
        rug_trace = RugTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            histogram_specifier=histogram_specifier,
        )
        traces["_".join((rug_trace.name, histogram_specifier.dimension))] = go.Scatter(
            rug_trace.model_dump()
        )

    if histogram_specifier.kde:
        kde_trace = KdeTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            histogram_specifier=histogram_specifier,
        )
        traces["_".join((kde_trace.name, histogram_specifier.dimension))] = go.Scatter(
            kde_trace.model_dump()
        )

    return traces
