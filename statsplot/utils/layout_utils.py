"""Utility functions to interact with layout objects."""

import re
from typing import Callable, Dict, Any, Set

from plotly import graph_objs as go

from statsplot import constants
from statsplot.plot_specifiers.data import DataHandler, SliceTraceType
from statsplot.plot_specifiers.trace import JointplotSpecifier, JointplotType


def slice_name_in_trace_name(slice_name: str) -> Callable:
    return re.compile(r"\b({0})\b".format(slice_name)).search


def adjust_jointplot_legends(
    jointplot_specifier: JointplotSpecifier,
    slices_marginal_traces: Dict[str, Any],
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
        if all(
            not histogram_specifier.hist
            for histogram_specifier in jointplot_specifier.histogram_specifier.values()
        ):
            legend_groups = []
            for trace in slices_marginal_traces:
                if (
                    legendgroup := slices_marginal_traces[trace].legendgroup
                ) not in legend_groups:
                    slices_marginal_traces[trace].update({"showlegend": True})
                    legend_groups.append(legendgroup)


def add_update_menu(
    fig: go.Figure,
    data_handler: DataHandler,
    slices_traces: Dict[str, Any] = {},
) -> go.Figure:
    trace_update_rule: Dict[str, Any] = {}

    # all data visibility rules
    trace_update_rule[SliceTraceType.ALL_DATA] = {
        "visibility": [
            trace.legendgroup == SliceTraceType.ALL_DATA
            or (
                trace.name
                in [slice_trace.name for slice_trace in slices_traces.values()]
            )
            for trace in fig.data
        ],
        "showlegend": [trace.showlegend for trace in fig.data],
        "legendgroup": [trace.legendgroup for trace in fig.data],
    }

    def set_and_update_visibility_status(trace_name: str) -> bool:
        if trace_name in visibility_set:
            return False
        visibility_set.add(trace_name)
        return True

    for level in data_handler.slice_levels:
        # slicer visibility rules
        visibility_set: Set[str] = set()
        trace_update_rule[level] = {
            "visibility": [
                slice_name_in_trace_name(level)(trace.name) is not None
                for trace in fig.data
            ],
            "showlegend": [
                set_and_update_visibility_status(trace.name) for trace in fig.data
            ],
            "legendgroup": [trace.name for trace in fig.data],
        }

    fig.update_layout(
        updatemenus=[
            dict(
                type=constants.LAYOUT_UPDATE_MENUS_TYPE,
                direction=constants.LAYOUT_UPDATE_MENUS_DIRECTION,
                active=0,
                x=1,
                y=1,
                buttons=list(
                    [
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
                    ]
                ),
            )
        ]
    )

    return fig


def smart_title(title_string: str) -> str:
    """Split string at _, capitalizes words, and joins with space."""
    title_string = title_string.strip()
    if len(title_string) == 0:
        return title_string
    return " ".join(
        [
            "".join([w[0].upper(), w[1:]])
            if (len(w) >= constants.MIN_CAPITALIZE_LENGTH)
            and not (any(l.isupper() for l in w))
            else w
            for w in re.split(" |_", title_string)
        ]
    )


def smart_legend(legend_string: str) -> str:
    """Cleans and capitalizes axis legends for figure."""
    legend_string = legend_string.strip()
    if len(legend_string) == 0:
        return legend_string
    return " ".join(
        [
            "".join([w[0].upper(), w[1:]]) if i == 0 else w
            for i, w in enumerate(re.split("_", legend_string))
        ]
    )
