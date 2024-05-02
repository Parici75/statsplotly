"""Utility functions to interact with layout objects."""

import re
from collections.abc import Callable
from typing import Any

from plotly import graph_objs as go

from statsplotly import constants
from statsplotly.plot_specifiers.data import DataHandler, SliceTraceType
from statsplotly.plot_specifiers.trace import JointplotSpecifier, JointplotType


def set_horizontal_colorbar(colorbar_dict: dict[str, Any]) -> dict[str, Any]:
    colorbar_dict.update(
        {
            "orientation": "h",
            "y": constants.COLORBAR_BOTTOM_POSITION,
            "xanchor": "center",
            "yanchor": "bottom",
        }
    )
    return colorbar_dict


def _slice_name_in_trace_name(slice_name: str) -> Callable[[str], re.Match[Any] | None]:
    return re.compile(rf"\b({slice_name})\b").search


def adjust_jointplot_legends(
    jointplot_specifier: JointplotSpecifier,
    slices_marginal_traces: dict[str, Any],
) -> None:
    if len(slices_marginal_traces) == 0:
        return

    if jointplot_specifier.plot_type in (
        JointplotType.SCATTER,
        JointplotType.SCATTER_KDE,
    ):
        for trace in slices_marginal_traces:
            slices_marginal_traces[trace].update({"showlegend": False})
    elif jointplot_specifier.histogram_specifier is not None:
        # Make sure legends are displayed
        if all(
            not histogram_specifier.hist
            for histogram_specifier in jointplot_specifier.histogram_specifier.values()
        ):
            legend_groups = []
            for trace in slices_marginal_traces:
                if (legendgroup := slices_marginal_traces[trace].legendgroup) not in legend_groups:
                    slices_marginal_traces[trace].update({"showlegend": True})
                    legend_groups.append(legendgroup)

        # Separate legend groups if we have only one slice
        if len(slices_marginal_traces) == 1:
            for trace in slices_marginal_traces:
                slices_marginal_traces[trace].update(
                    {
                        "legendgroup": " ".join(
                            (slices_marginal_traces[trace].legendgroup, "marginal")
                        )
                    }
                )


def add_update_menu(
    fig: go.Figure,
    data_handler: DataHandler,
    slices_traces: dict[str, Any] | None = None,
    preplotted_traces: dict[str, Any] | None = None,
) -> go.Figure:
    trace_update_rule: dict[str, Any] = {}
    if slices_traces is None:
        slices_traces = {}
    if preplotted_traces is None:
        preplotted_traces = {}

    # all data visibility rules
    if SliceTraceType.ALL_DATA.value in [trace.legendgroup for trace in fig.data]:
        trace_update_rule[SliceTraceType.ALL_DATA.value] = {
            "visibility": [
                trace.legendgroup == SliceTraceType.ALL_DATA.value
                or (
                    trace.name
                    in [
                        slice_trace.name
                        for slice_trace in {**slices_traces, **preplotted_traces}.values()
                    ]
                )
                for trace in fig.data
            ],
            "showlegend": [trace.showlegend for trace in fig.data],
            "legendgroup": [
                (
                    trace.legendgroup
                    if trace.name
                    not in [slice_trace.name for slice_trace in slices_traces.values()]
                    else SliceTraceType.SLICE.value
                )
                for trace in fig.data
            ],
        }

    def set_and_update_visibility_status(trace_name: str) -> bool:
        if trace_name in visibility_set:
            return False
        visibility_set.add(trace_name)
        return True

    for level in data_handler.slice_levels:
        # slicer visibility rules
        visibility_set: set[str] = set()
        trace_update_rule[level] = {
            "visibility": [
                _slice_name_in_trace_name(level)(trace.name) is not None for trace in fig.data
            ],
            "showlegend": [set_and_update_visibility_status(trace.name) for trace in fig.data],
            "legendgroup": [trace.name for trace in fig.data],
        }

    # Update layout
    fig.update_layout(
        updatemenus=[
            {
                "type": constants.LAYOUT_UPDATE_MENUS_TYPE,
                "direction": constants.LAYOUT_UPDATE_MENUS_DIRECTION,
                "active": 0,
                "x": 1,
                "y": 1,
                "buttons": [
                    {
                        "label": f"{data_handler.data_pointer.slicer}: {level}",
                        "method": "update",
                        "args": [
                            {
                                "visible": trace_update["visibility"],
                                "showlegend": trace_update["showlegend"],
                                "legendgroup": trace_update["legendgroup"],
                            }
                        ],
                    }
                    for level, trace_update in trace_update_rule.items()
                ],
            }
        ]
    )

    # Adjust initial parameters
    for i, trace in enumerate(fig.data):
        trace.update(
            {
                parameter: fig.layout.updatemenus[0]["buttons"][0]["args"][0][parameter][i]
                for parameter in ["visible", "showlegend", "legendgroup"]
            }
        )

    return fig
