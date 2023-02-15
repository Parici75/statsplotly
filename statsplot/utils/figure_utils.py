"""Figure utilities."""

import logging
import os
import re
from typing import Dict, List, Tuple

import plotly.graph_objs as go

from pymodules.file_utils import mkdir, process_filename
from statsplot.plot_objects.layout_objects import XYLayout, SceneLayout
from statsplot.plot_objects.trace_objects import BaseTrace

logger = logging.getLogger(__name__)


def create_fig(
    fig: go.Figure,
    traces: Dict[str, BaseTrace],
    layout: XYLayout | SceneLayout,
    row: int,
    col: int,
    secondary_y: bool = False,
) -> go.Figure:
    """Creates or updates a figure with the appropriate layout."""
    layout_dict = layout.dict()
    if fig is None:
        return go.Figure(
            data=list(traces.values()),
            layout=go.Layout(layout_dict),
        )
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
            if (
                axis_number_pattern := re.search(r"\d+", axis_ref)
            ) is not None:
                axis_number = axis_number_pattern.group()
                if (
                    scaleanchor := layout_dict[axis_ref].get("scaleanchor")
                ) is not None:
                    scaleanchor_root = re.sub(
                        r"\d", axis_number_pattern.group(), scaleanchor
                    )
                    layout_dict[axis_ref].update(
                        {"scaleanchor": f"{scaleanchor_root}{axis_number}"}
                    )

        # Remove axes titles
        if row < len(fig._grid_ref):
            layout_dict[xaxis_ref]["title"] = None
        if col > 1:
            layout_dict[yaxis_ref]["title"] = None

    # Update layout
    fig.update_layout(**layout_dict)

    return fig


def clean_subplots(
    fig: go.Figure,
    title: str = None,
    no_legend: bool = False,
    clean_yaxes_title: bool = True,
    row_titles: List | None = None,
    col_titles: List | None = None,
) -> Tuple[go.Figure, str]:
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
            lambda trace: trace.update(showlegend=False)
            if (trace.name in names)
            else names.add(trace.name)
        )

    # Y axes
    if clean_yaxes_title:
        for row, subplot_row in enumerate(fig._grid_ref):
            if row < len(fig._grid_ref) - 1:
                for subplot in subplot_row:
                    xaxis, yaxis = subplot[0][1]
                    fig.update_layout({yaxis: {"title": None}})

    if col_titles is not None:
        for i, col_title in enumerate(col_titles, 1):
            x_unit = 1 / len(fig._grid_ref[0])
            fig.add_annotation(
                {
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
                }
            )

    if row_titles is not None:
        for i, row_title in enumerate(row_titles[::-1], 1):
            y_unit = 1 / len(fig._grid_ref)
            fig.add_annotation(
                {
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
                }
            )
        # Add some left margin
        try:
            fig.update_layout({"margin": fig.layout.margin.l + 10})
        except TypeError:
            fig.layout.margin.l = 150

    return fig, title


def save(
    fig: go.Figure,
    title: str | None = None,
    plot_dir: str | None = None,
    pdf_write: bool = False,
    png_write: bool = False,
) -> None:
    """Saves the figure to various file format."""

    if title is None:
        title = fig.layout.title.text or "plotly_figure"

    if plot_dir is None:
        path = os.getcwd()
    else:
        path = mkdir(plot_dir)

    filename = process_filename(title)
    fig.write_html(os.path.join(path, filename + ".html"))
    if pdf_write:
        pdf_dir = mkdir(os.path.join(path, "pdfs"))
        fig.write_image(os.path.join(pdf_dir, filename + ".pdf"))
        logger.info(f"Saved {filename} to {pdf_dir}")
    if png_write:
        png_dir = mkdir(os.path.join(path, "pngs"))
        fig.write_image(os.path.join(png_dir, filename + ".png"))
        logger.info(f"Saved {filename} to {png_dir}")
