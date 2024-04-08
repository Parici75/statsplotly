import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots
from pydantic import ValidationError

from statsplotly import constants, heatmap
from statsplotly.exceptions import StatsPlotSpecificationError
from statsplotly.plot_objects.layout import ScatterLayout
from statsplotly.plot_objects.layout._axis import ColorAxis
from statsplotly.plot_objects.trace import ScatterTrace
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.plot_specifiers.figure import create_fig
from statsplotly.plot_specifiers.figure._utils import (
    FigureSubplotFormatter,
    SharedGridAxis,
    SubplotGridFormatter,
)
from statsplotly.plot_specifiers.layout import AxesSpecifier


@pytest.fixture
def example_figure_layout_traces():
    fig = make_subplots(rows=2, cols=2)
    fig.add_trace(
        go.Scatter(x=[1, 2, 3], y=[4, 5, 6], marker=dict(color=[7, 8, 9], coloraxis="coloraxis"))
    )
    fig.update_layout({"coloraxis": {}, "coloraxis2": {}})

    layout_dict = {"coloraxis": {}}
    traces = {
        "new_trace_with_coloraxis": go.Scatter(
            x=[4, 5, 6], y=[7, 8, 9], marker=dict(color=[10, 11, 12], coloraxis="coloraxis")
        ),
        "new_trace_without_coloraxis": go.Scatter(
            x=[7, 8, 9], y=[10, 11, 12], marker=dict(color=[13, 14, 15])
        ),
    }
    return fig, layout_dict, traces


@pytest.fixture
def example_heatmap_subplots():
    data = pd.concat(
        [
            pd.DataFrame({"x": np.arange(5), "y": np.arange(5) - 2, "z": np.arange(5) * i})
            for i in np.arange(4)
        ],
        keys=np.arange(4),
    )
    nrows, ncols = (2, 2)
    fig = make_subplots(rows=nrows, cols=ncols)
    for i, j in itertools.product(np.arange(nrows) + 1, np.arange(nrows) + 1):
        fig = heatmap(
            fig=fig,
            row=i,
            col=j,
            data=data.loc[i + j - 1],
            x="x",
            y="y",
            z="z",
        )

    return fig


@pytest.fixture
def example_figure_data(example_trace_data, example_legend):
    # Create a trace
    trace = go.Scatter(
        **ScatterTrace.build_trace(
            trace_data=example_trace_data,
            trace_name="A",
            trace_color="red",
            color_specifier=ColorSpecifier(),
            mode="markers",
        ).model_dump()
    )

    # Create a layout
    layout = ScatterLayout.build_layout(
        axes_specifier=AxesSpecifier(traces=[example_trace_data], legend=example_legend),
        coloraxis=ColorAxis(),
    )

    return {trace.name: trace}, layout


class TestCreateFig:

    def test_create_fig(self, example_figure_data):
        traces, layout = example_figure_data
        # Create a figure with traces and layout
        fig = create_fig(fig=None, traces=traces, layout=layout, row=1, col=1)

        # Assert that the figure contains the expected traces and layout
        assert len(fig.data) == 1
        assert fig.layout.title.text == "y vs x"

    def test_create_fig_on_second_row(self, example_figure_data):
        traces, layout = example_figure_data
        # Create a figure with traces and layout
        fig = create_fig(fig=None, traces=traces, layout=layout, row=2, col=1)

        # Assert that the figure contains the expected traces and layout
        assert len(fig._grid_ref) == 2

    def test_error_on_create_subplot(self, example_figure_data):
        traces, layout = example_figure_data
        # Trying to add a plot to a figure object without a subplot grid raises a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            create_fig(fig=go.Figure(), traces=traces, layout=layout, row=1, col=1)

    def test_add_fig_to_existing_layout(self, example_figure_data):
        traces, layout = example_figure_data
        # Create a figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig = create_fig(fig=fig, traces=traces, layout=layout, row=1, col=1)
        assert len(fig.data) == 1

        # Update the figure layout with new traces and layout
        fig = create_fig(fig=fig, traces=traces, layout=layout, row=2, col=1)

        # Assert that the figure contains the expected traces and updated layout
        assert len(fig.data) == 2
        assert "xaxis2" in fig.layout and "yaxis2" in fig.layout
        assert fig.layout.xaxis.title.text == None
        assert fig.layout.xaxis2.title.text == "X"


def test_update_coloraxis_reference(example_figure_layout_traces):
    fig, layout_dict, traces = example_figure_layout_traces
    figure_layout_formatter = FigureSubplotFormatter(fig=fig, row=1, col=1)
    figure_layout_formatter.update_coloraxis_reference(
        coloraxis_reference="coloraxis",
        layout_dict=layout_dict,
        traces=traces,
    )
    assert traces["new_trace_with_coloraxis"].marker.coloraxis == "coloraxis3"
    assert (
        len([key for key in traces["new_trace_without_coloraxis"] if key.startswith("coloraxis")])
        == 0
    )


def test_format_colorbar_in_subplots_layout(example_figure_layout_traces):
    fig, layout_dict, traces = example_figure_layout_traces

    # Update layout with some colorbars
    layout_dict.update(
        {
            "coloraxis": {"colorbar": {"y": 0.5, "yanchor": "middle", "len": 1}},
            "coloraxis2": {"colorbar": {"x": 0.5, "xanchor": "left", "len": 1}},
        }
    )

    # Adding plot to the second row
    FigureSubplotFormatter(fig=fig, row=2, col=1).format_colorbar_in_subplots_layout(
        layout_dict["coloraxis"]
    )
    assert layout_dict["coloraxis"]["colorbar"]["y"] == 0
    assert np.allclose(layout_dict["coloraxis"]["colorbar"]["len"], 0.425)
    assert layout_dict["coloraxis"]["colorbar"]["yanchor"] == "bottom"

    # Adding plot to the second column
    FigureSubplotFormatter(fig=fig, row=1, col=2).format_colorbar_in_subplots_layout(
        layout_dict["coloraxis2"]
    )
    assert layout_dict["coloraxis2"]["colorbar"]["x"] == 1.0
    assert np.allclose(layout_dict["coloraxis2"]["colorbar"]["len"], 0.425)
    assert layout_dict["coloraxis2"]["colorbar"]["xanchor"] == "left"
    assert layout_dict["coloraxis2"]["colorbar"]["y"] == 0.575
    assert np.allclose(layout_dict["coloraxis2"]["colorbar"]["thickness"], 13.5)


class TestSetCommonColoraxis:
    def test_all_common_coloraxis(self, example_heatmap_subplots):
        SubplotGridFormatter(fig=example_heatmap_subplots).set_common_coloraxis(SharedGridAxis.ALL)
        assert all([trace.coloraxis == "coloraxis5" for trace in example_heatmap_subplots.data])
        assert example_heatmap_subplots.layout["coloraxis5"]["cmin"] == 0
        assert example_heatmap_subplots.layout["coloraxis5"]["cmax"] == 12

    def test_columns_common_coloraxis(self, example_heatmap_subplots):
        # Set common color axis for columns
        SubplotGridFormatter(fig=example_heatmap_subplots).set_common_coloraxis(SharedGridAxis.COLS)
        assert all(
            [
                trace.coloraxis in ["coloraxis4", "coloraxis5"]
                for trace in example_heatmap_subplots.data
            ]
        )
        assert example_heatmap_subplots.layout["coloraxis4"]["cmin"] == 0
        assert example_heatmap_subplots.layout["coloraxis4"]["cmax"] == 8
        assert example_heatmap_subplots.layout["coloraxis4"]["colorbar"]["orientation"] == "h"
        assert example_heatmap_subplots.layout["coloraxis4"]["colorbar"]["x"] == 0.45
        assert example_heatmap_subplots.layout["coloraxis4"]["colorbar"]["y"] == -0.5
        assert example_heatmap_subplots.layout["coloraxis4"]["colorbar"]["xanchor"] == "right"
        assert np.allclose(example_heatmap_subplots.layout["coloraxis4"]["colorbar"]["len"], 0.425)
        assert example_heatmap_subplots.layout["coloraxis5"]["cmin"] == 0
        assert example_heatmap_subplots.layout["coloraxis5"]["cmax"] == 12

    def test_rows_common_coloraxis(self, example_heatmap_subplots):
        # Set common color axis for rows
        SubplotGridFormatter(fig=example_heatmap_subplots).set_common_coloraxis(SharedGridAxis.ROWS)
        assert all(
            [
                trace.coloraxis in ["coloraxis3", "coloraxis5"]
                for trace in example_heatmap_subplots.data
            ]
        )
        assert example_heatmap_subplots.layout["coloraxis3"]["cmin"] == 0
        assert example_heatmap_subplots.layout["coloraxis3"]["cmax"] == 8
        assert (
            example_heatmap_subplots.layout["coloraxis3"]["colorbar"]["thickness"]
            == constants.COLORBAR_THICKNESS_SCALING_FACTOR
        )
        assert example_heatmap_subplots.layout["coloraxis5"]["cmin"] == 0
        assert example_heatmap_subplots.layout["coloraxis5"]["cmax"] == 12


class TestSetCommonAxisLimits:
    def test_all_common_axis_limits(self, example_heatmap_subplots):
        SubplotGridFormatter(fig=example_heatmap_subplots).set_common_axis_limit(
            shared_grid_axis=SharedGridAxis.ALL, plot_axis="xaxis"
        )
        assert all(
            example_heatmap_subplots.layout[layout_key]["range"] == (0.0, 4.4)
            for layout_key in example_heatmap_subplots.layout
            if "xaxis" in layout_key
        )

        with pytest.raises(ValidationError) as excinfo:
            SubplotGridFormatter(fig=example_heatmap_subplots).set_common_axis_limit(
                shared_grid_axis=SharedGridAxis.ALL
            )
            assert (
                f"`plot_axis` must be specified when using `shared_grid_axis = {SharedGridAxis.ALL.value}`"
                in str(excinfo.value)
            )

    def test_columns_common_axis_limits(self, example_heatmap_subplots):
        SubplotGridFormatter(fig=example_heatmap_subplots).set_common_axis_limit(
            shared_grid_axis="cols", plot_axis="yaxis"
        )
        assert all(
            example_heatmap_subplots.layout[layout_key]["range"] == (-2.2, 2.2)
            for layout_key in example_heatmap_subplots.layout
            if "yaxis" in layout_key
        )

    def test_linked_yaxis_rows_common_axis_limits(self, example_heatmap_subplots):
        SubplotGridFormatter(fig=example_heatmap_subplots).set_common_axis_limit(
            shared_grid_axis="rows", link_axes=True
        )
        assert all(
            example_heatmap_subplots.layout[layout_key]["matches"] is None
            for layout_key in example_heatmap_subplots.layout
            if "xaxis" in layout_key
        )
        assert all(
            example_heatmap_subplots.layout[layout_key]["matches"] is not None
            for layout_key in example_heatmap_subplots.layout
            if "yaxis" in layout_key
        )
        assert example_heatmap_subplots.layout["yaxis2"]["matches"] == "y2"


class TestTidySubplots:
    def test_update_title(self, example_heatmap_subplots):
        SubplotGridFormatter(fig=example_heatmap_subplots).tidy_subplots(title="new_title")
        assert example_heatmap_subplots.layout.title.text == "new_title"

    def test_row_titles(self, example_heatmap_subplots):
        SubplotGridFormatter(fig=example_heatmap_subplots).tidy_subplots(row_titles=["1", "2"])
        assert len(example_heatmap_subplots.layout.annotations) == 2
        assert example_heatmap_subplots.layout.annotations[0]["x"] == 0
        assert example_heatmap_subplots.layout.annotations[0]["y"] == 0.7875
