"""Figure utilities."""

import logging
import re

import plotly.graph_objs as go

from statsplotly.plot_objects.layout_objects import SceneLayout, XYLayout
from statsplotly.plot_objects.trace_objects import BaseTrace
from statsplotly.plot_specifiers.layout import ColoraxisReference

logger = logging.getLogger(__name__)


def increment_axis_reference(axis_reference: str) -> str:
    splitted_ref = re.split(r"(\d+)", axis_reference)
    if len(splitted_ref) == 1:
        return "".join((axis_reference, "2"))

    return "".join((splitted_ref[0], str(int(splitted_ref[1]) + 1)))


def create_fig(  # noqa: PLR0912 PLR0913 C901
    fig: go.Figure,
    traces: dict[str, BaseTrace],
    layout: XYLayout | SceneLayout,
    row: int,
    col: int,
    secondary_y: bool = False,
) -> go.Figure:
    """Creates or updates a figure with the appropriate layout."""
    layout_dict = layout.model_dump()
    if fig is None:
        return go.Figure(
            data=list(traces.values()),
            layout=go.Layout(layout_dict),
        )

    # Handle multiple coloraxis
    if ColoraxisReference.MAIN_COLORAXIS in layout_dict:
        if (
            len(
                coloraxis_refs := sorted(
                    [key for key in fig.layout if key.startswith("coloraxis")],
                    reverse=True,
                )
            )
            > 0
        ):
            new_coloraxis_ref = increment_axis_reference(coloraxis_refs[0])
            layout_dict[new_coloraxis_ref] = layout_dict.pop(ColoraxisReference.MAIN_COLORAXIS)
            # Update traces objects
            for trace in traces.values():
                if hasattr(trace, "coloraxis"):
                    if trace.coloraxis is not None:
                        trace.coloraxis = new_coloraxis_ref

                if hasattr(trace, "marker"):
                    if hasattr(trace.marker, "coloraxis"):
                        if trace.marker["coloraxis"] is not None:
                            trace.marker = trace.marker.update({"coloraxis": new_coloraxis_ref})

    # Add the new traces
    for trace in traces.values():
        fig.add_trace(trace, row=row, col=col, secondary_y=secondary_y)

    # Rename layout axes keys to match position in the layout
    if isinstance(layout_dict, SceneLayout):
        scene = fig._grid_ref[row - 1][col - 1][0][1][0]
        layout_dict[scene] = layout_dict.pop("scene")

    else:
        # Normal plot
        axis = fig._grid_ref[row - 1][col - 1]
        if secondary_y:
            xaxis_ref, yaxis_ref = axis[1][1]
        else:
            # Extract xaxis and yaxis axes
            xaxis_ref, yaxis_ref = axis[0][1]
        # Update layout
        layout_dict[xaxis_ref] = layout_dict.pop("xaxis")
        layout_dict[yaxis_ref] = layout_dict.pop("yaxis")
        # Rename axes references
        for axis_ref in [xaxis_ref, yaxis_ref]:
            if (axis_number_pattern := re.search(r"\d+", axis_ref)) is not None:
                axis_number = axis_number_pattern.group()
                if (scaleanchor := layout_dict[axis_ref].get("scaleanchor")) is not None:
                    scaleanchor_root = re.sub(r"\d", axis_number_pattern.group(), scaleanchor)
                    layout_dict[axis_ref].update({
                        "scaleanchor": f"{scaleanchor_root}{axis_number}"
                    })

        # Remove axes titles
        if row < len(fig._grid_ref):
            layout_dict[xaxis_ref]["title"] = None
        if col > 1:
            layout_dict[yaxis_ref]["title"] = None

    # Update layout
    fig.update_layout({key: value for key, value in layout_dict.items() if value is not None})

    return fig


def _clean_col_titles(titles: list[str], fig: go.Figure) -> go.Figure:
    for i, col_title in enumerate(titles, 1):
        x_unit = 1 / len(fig._grid_ref[0])
        fig.add_annotation({
            "font": {"size": 16},
            "showarrow": False,
            "text": col_title,
            "x": x_unit * i - x_unit / 2,
            "xanchor": "center",
            "xref": "paper",
            "y": 1,
            "yanchor": "top",
            "yref": "paper",
            "yshift": +30,
        })
    return fig


def _clean_row_titles(titles: list[str], fig: go.Figure) -> go.Figure:
    for i, row_title in enumerate(titles[::-1], 1):
        y_unit = 1 / len(fig._grid_ref)
        fig.add_annotation({
            "font": {"size": 16},
            "showarrow": False,
            "text": row_title,
            "x": 0,
            "textangle": 0,
            "xanchor": "right",
            "xref": "paper",
            "xshift": -40,
            "y": y_unit * i - y_unit / 2,
            "yanchor": "middle",
            "yref": "paper",
        })

    # Add some left margin
    try:
        fig.update_layout({"margin": fig.layout.margin.l + 10})
    except TypeError:
        fig.layout.margin.l = 150

    return fig


def clean_subplots(  # noqa: PLR0913
    fig: go.Figure,
    title: str | None = None,
    no_legend: bool = False,
    clean_yaxes_title: bool = True,
    row_titles: list[str] | None = None,
    col_titles: list[str] | None = None,
) -> tuple[go.Figure, str]:
    """Cleans subplots of extra titles and legend."""

    # Replace title if supplied
    if title is not None:
        fig.update_layout(title=title)
    else:
        title = fig.layout.title.text

    # Clean legend
    if no_legend:
        fig.update_layout({"showlegend": False})
    else:
        # Remove legend title
        fig.update_layout({"legend": {"title": {"text": ""}}})
        # Remove legend duplicates
        names = set()
        fig.for_each_trace(
            lambda trace: (
                trace.update(showlegend=False) if (trace.name in names) else names.add(trace.name)
            )
        )

    # Y axes
    if clean_yaxes_title:
        for row, subplot_row in enumerate(fig._grid_ref):
            if row < len(fig._grid_ref) - 1:
                for subplot in subplot_row:
                    xaxis, yaxis = subplot[0][1]
                    fig.update_layout({yaxis: {"title": None}})

    if col_titles is not None:
        fig = _clean_col_titles(col_titles, fig)

    if row_titles is not None:
        fig = _clean_row_titles(row_titles, fig)

    return fig, title
