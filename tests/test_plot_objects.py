import numpy as np
import pytest

from statsplotly.plot_objects.trace import (
    BaseTrace,
    HeatmapTrace,
    HistogramTrace,
    KdeTrace,
    Scatter3DTrace,
    ScatterTrace,
)
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.plot_specifiers.data import DataDimension, HistogramNormType
from statsplotly.plot_specifiers.trace import HistogramSpecifier, TraceMode

TRACE_NAME = "dummy_name"


def test_base_trace(example_trace_data):
    with pytest.raises(TypeError):
        BaseTrace(x=example_trace_data.x_values, y=example_trace_data.y_values, name="dummy_name")


def test_heatmap_trace(example_3dtrace_data):
    heatmap_trace = HeatmapTrace.build_trace(
        trace_data=example_3dtrace_data,
        trace_name=TRACE_NAME,
        color_specifier=ColorSpecifier(),
    )
    assert all(heatmap_trace.x == example_3dtrace_data.x_values)
    assert all(heatmap_trace.y == example_3dtrace_data.y_values)
    assert all(heatmap_trace.z == example_3dtrace_data.z_values)
    assert heatmap_trace.coloraxis is None
    assert heatmap_trace.zmin is None
    assert heatmap_trace.zmax is None
    assert heatmap_trace.text is None
    assert heatmap_trace.colorbar is None
    assert heatmap_trace.colorscale is None


def test_scatter_trace(example_trace_data):
    scatter_trace = ScatterTrace.build_trace(
        trace_data=example_trace_data,
        trace_name=TRACE_NAME,
        trace_color=None,
        color_specifier=ColorSpecifier(),
        mode=None,
    )
    assert all(scatter_trace.x == example_trace_data.x_values)
    assert all(scatter_trace.y == example_trace_data.y_values)
    assert all(scatter_trace.text == example_trace_data.text_data)
    assert scatter_trace.mode is None
    assert scatter_trace.name == TRACE_NAME
    assert scatter_trace.legendgroup == TRACE_NAME
    assert scatter_trace.showlegend is None
    assert scatter_trace.marker == {
        "size": None,
        "color": None,
        "opacity": None,
        "symbol": None,
        "coloraxis": None,
    }
    assert scatter_trace.error_x is None
    assert scatter_trace.error_y is None
    assert scatter_trace.get_error_bars(example_trace_data) == [None] * 3


def test_scatter3d_trace(example_3dtrace_data):
    scatter_trace = Scatter3DTrace.build_trace(
        trace_data=example_3dtrace_data,
        trace_name=TRACE_NAME,
        trace_color="blue",
        color_specifier=ColorSpecifier(),
        mode="lines",
    )
    assert all(scatter_trace.x == example_3dtrace_data.x_values)
    assert all(scatter_trace.y == example_3dtrace_data.y_values)
    assert all(scatter_trace.z == example_3dtrace_data.z_values)
    assert scatter_trace.text is None
    assert scatter_trace.mode is TraceMode.LINES
    assert scatter_trace.error_z is None
    assert scatter_trace.marker == {
        "size": None,
        "color": "blue",
        "opacity": None,
        "symbol": None,
        "coloraxis": None,
        "line": {"color": "grey", "width": 2},
    }


def test_histogram_trace(example_trace_data):
    histogram_trace = HistogramTrace.build_trace(
        trace_data=example_trace_data,
        trace_name=TRACE_NAME,
        trace_color=None,
        color_specifier=ColorSpecifier(),
        histogram_specifier=HistogramSpecifier(
            hist=True,
            dimension=DataDimension.Y,
            histnorm="",
            bins=10,
            data_type=np.dtype("int"),
        ),
    )
    assert histogram_trace.x is None
    assert all(histogram_trace.y == example_trace_data.y_values)
    assert histogram_trace.name == "dummy_name distribution"
    assert histogram_trace.opacity is None
    assert histogram_trace.legendgroup == TRACE_NAME
    assert histogram_trace.showlegend is None
    assert histogram_trace.marker == {"color": None}
    assert histogram_trace.cumulative == {"enabled": None}
    assert histogram_trace.xbins == {"start": 0.0, "end": 2.0, "size": 0.2}
    assert histogram_trace.histnorm is HistogramNormType.COUNT


def test_kde_trace(example_trace_data):
    kde_trace = KdeTrace.build_trace(
        trace_data=example_trace_data,
        trace_name=TRACE_NAME,
        trace_color=None,
        color_specifier=ColorSpecifier(),
        histogram_specifier=HistogramSpecifier(
            hist=True,
            dimension=DataDimension.Y,
            histnorm="",
            bins=10,
            data_type=np.dtype("int"),
        ),
    )
    assert kde_trace.hoverinfo == "x+y"
