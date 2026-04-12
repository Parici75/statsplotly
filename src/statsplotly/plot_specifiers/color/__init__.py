"""This subpackage defines objects and utility methods for color properties."""

from ._core import ColorSpecifier, HistogramColorSpecifier
from ._utils import rgb_string_array_from_colormap, set_rgb_alpha

__all__ = [
    "ColorSpecifier",
    "HistogramColorSpecifier",
    "rgb_string_array_from_colormap",
    "set_rgb_alpha",
]
