"""This subpackage defines objects and utility methods for color properties."""

from ._core import ColorSpecifier, HistogramColorSpecifier
from ._utils import set_rgb_alpha

__all__ = ["ColorSpecifier", "HistogramColorSpecifier", "set_rgb_alpha"]
