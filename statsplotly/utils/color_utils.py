import logging
import re
from enum import Enum
from typing import Any, TypeAlias

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import seaborn as sns
from numpy.typing import NDArray

from statsplotly import constants
from statsplotly.exceptions import StatsPlotSpecificationError, UnsupportedColormapError

logger = logging.getLogger(__name__)

Cmap_specs: TypeAlias = str | list[str] | list[tuple[float, float, float]]


class ColorSystem(str, Enum):
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    DISCRETE = "discrete"


def rand_jitter(arr: NDArray[Any], jitter_amount: float = 1) -> NDArray[Any]:
    """from https://stackoverflow.com/questions/8671808/matplotlib-avoiding
    -overlapping-datapoints-in-a-scatter-dot-beeswarm-plot"""
    spread = 0.01 * (max(arr) - min(arr)) * jitter_amount or 0.05 * jitter_amount

    return arr + np.random.randn(len(arr)) * spread


def get_colorarray_from_seaborn(cmap: Cmap_specs | None, n_colors: int) -> NDArray[Any]:
    try:
        return sns.color_palette(cmap, n_colors=n_colors)
    except ValueError as exc:
        if isinstance(cmap, str):
            return sns.color_palette(cmap.lower(), n_colors=n_colors)
        raise exc


def get_colorarray_from_matplotlib(
    cmap: Cmap_specs | None,
    n_colors: int,
) -> NDArray[Any]:
    if isinstance(cmap, list):
        raise StatsPlotSpecificationError("Colormap specification is not matplotlib-compatible")
    try:
        mpl_cmap = plt.get_cmap(cmap, n_colors)
    except ValueError as exc:
        if isinstance(cmap, str):
            mpl_cmap = plt.get_cmap(cmap.lower(), n_colors)
        raise exc
    return mpl_cmap(np.linspace(0, 1, n_colors))


def get_colorarray_from_plotly(cmap: Cmap_specs, n_colors: int) -> NDArray[Any]:
    if n_colors == 1:
        return plotly.colors.sample_colorscale(cmap, samplepoints=0.5, colortype="tuple")

    return plotly.colors.sample_colorscale(cmap, samplepoints=n_colors, colortype="tuple")


def cmap_to_array(
    n_colors: int,
    cmap: Cmap_specs | matplotlib.colors.Colormap | None,
) -> NDArray[Any]:
    """Returns n_colors linearly spaced values on the colormap specified from cmap."""

    # If cmap is a matplotlib colormap, get color values from it
    if isinstance(cmap, matplotlib.colors.Colormap):
        return cmap(np.linspace(0, 1, n_colors))

    # Plotly
    if cmap is not None:
        try:
            return get_colorarray_from_plotly(cmap, n_colors=n_colors)
        except Exception as exc:
            logger.debug(f"Plotly error processing {cmap} colormap: \n{exc}")

    try:
        # Try seaborn first
        return get_colorarray_from_seaborn(cmap, n_colors=n_colors)
    except ValueError:
        try:
            # Then matplotlib
            return get_colorarray_from_matplotlib(cmap, n_colors=n_colors)
        except ValueError as exc:
            raise UnsupportedColormapError(f"{cmap} is not a supported colormap") from exc


def to_rgb(numeric_array: NDArray[Any]) -> list[str]:
    """Transforms a list of numeric rgb tuple into a list of rgb strings"""
    rgb_array = [
        "rgb" + str(rgb_tuple[:3])
        for rgb_tuple in [tuple(int(rgb * 256) for rgb in x) for x in numeric_array]
    ]
    return rgb_array


def get_rgb_discrete_array(
    n_colors: int, color_palette: Cmap_specs | matplotlib.colors.Colormap | None
) -> list[str]:
    """Color list/Seaborn color_palette wrapper."""
    rgb_array = cmap_to_array(n_colors, color_palette)

    # Convert the RGB value array to a RGB plotly_friendly string array
    return to_rgb(rgb_array)


def compute_colorscale(  # noqa PLR0912 C901
    n_colors: int,
    color_system: ColorSystem,
    logscale: float | None = 10,
    color_palette: Cmap_specs | matplotlib.colors.Colormap | None = None,
) -> str | list[list[float | str]]:
    """Returns a plotly-compatible colorscale depending on color system
    chosen by user."""

    if color_palette is None and color_system in (
        ColorSystem.LINEAR,
        ColorSystem.LOGARITHMIC,
    ):
        color_palette = sns.color_palette(
            palette=constants.SEABORN_DEFAULT_CONTINUOUS_COLOR_PALETTE,
            as_cmap=True,
        )

    # If color_palette is a list, then construct a colormap from it
    if isinstance(color_palette, list):
        color_palette = matplotlib.colors.LinearSegmentedColormap.from_list("", color_palette)

    if color_system is ColorSystem.LOGARITHMIC:
        if logscale is None:
            raise ValueError(
                f"Logscale can not be `None` when using {ColorSystem.LOGARITHMIC} color system."
            )
        # Get the actual colors from the colormap
        try:
            colors = cmap_to_array(n_colors, color_palette)
        except UnsupportedColormapError as exc:
            raise ValueError(
                f"{color_palette} is not supported for {color_system.value} color mapping, please "
                "specify a Matplotlib-supported colormap"
            ) from exc
        else:
            color_palette = to_rgb(colors)

        nsample = len(color_palette)
        colorscale = []
        for log_scale, color_index in zip(
            np.logspace(-1, 0, nsample, base=logscale),
            [int(x) for x in np.linspace(0, n_colors - 1, nsample)],
            strict=True,
        ):
            colorscale.append([log_scale, color_palette[color_index]])
        # Plotly wants the first index of the colorscale to 0 and last to 1
        colorscale[0][0] = 0
        colorscale[-1][0] = 1

    elif color_system is ColorSystem.LINEAR:
        if isinstance(color_palette, str):
            if color_palette.lower() in constants.BUILTIN_COLORSCALES:
                return color_palette
        try:
            colors = cmap_to_array(n_colors, color_palette)
        except UnsupportedColormapError as exc:
            raise ValueError(
                f"{color_palette} is not supported for {color_system.value} mapping, please "
                "specify a plotly or matplotlib-supported colorscale"
            ) from exc
        else:
            color_palette = to_rgb(colors)
        nsample = len(color_palette)
        colorscale = []
        for lin_scale, color_index in zip(
            np.linspace(0, 1, nsample),
            [int(x) for x in np.linspace(0, n_colors - 1, nsample)],
            strict=True,
        ):
            colorscale.append([lin_scale, color_palette[color_index]])

    elif color_system is ColorSystem.DISCRETE:
        try:
            colors = cmap_to_array(n_colors, color_palette)
            color_palette = to_rgb(colors)
        except UnsupportedColormapError as exc:
            raise ValueError(
                f"{color_palette} is not supported for {color_system.value} mapping, please "
                "specify a matplotlib-supported colorscale"
            ) from exc
        # Initialize colormap
        colorscale = []
        # We need to specify boundaries for the colormap. We repeat colormap
        # internal indices, and tile the color_palette
        for map_index, color in zip(
            np.tile(np.array([np.linspace(0, 1, n_colors + 1)]).T, 2).ravel()[1:-1],
            np.tile(np.array([color_palette]).T, 2).ravel(),
            strict=True,
        ):
            colorscale.append([map_index, color])

    return colorscale


def set_rgb_alpha(color_string: str, alpha: float | None = 1) -> str:
    """Transforms a rgb string into a rgba string or adjust alpha value."""
    if re.search("rgba", color_string) is not None:
        # Changing alpha value
        rgba_string = re.sub(r"\d.\d*", str(alpha), color_string)
    else:
        # Converting to rgb to rgba string
        rgba_string = f"{re.sub('rgb', 'rgba', color_string)[:-1]} , {str(alpha)})"

    return rgba_string
