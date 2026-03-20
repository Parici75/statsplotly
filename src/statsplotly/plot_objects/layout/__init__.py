from typing import TypeAlias

from ._axis import ColorAxis
from ._layout import (
    BarLayout,
    CategoricalLayout,
    HeatmapLayout,
    HistogramLayout,
    ScatterLayout,
    SceneLayout,
)

__all__ = [
    "ColorAxis",
    "BarLayout",
    "CategoricalLayout",
    "HeatmapLayout",
    "HistogramLayout",
    "ScatterLayout",
    "SceneLayout",
]

layout_type: TypeAlias = (
    HeatmapLayout | CategoricalLayout | ScatterLayout | SceneLayout | BarLayout | HistogramLayout
)
