"""Custom exceptions."""

from enum import Enum
from typing import Any


class StatsPlotInvalidArgumentError(ValueError):
    """Raises when argument can not be parsed with the corresponding enum."""

    def __init__(self, value: Any, enum: Enum) -> None:
        message = (
            f"Invalid value: '{value}'. Value must be one of"
            f" {[member.value for member in enum]}"  # type: ignore
        )
        super().__init__(message)


class StatsPlotSpecificationError(ValueError):
    """Raises when plot arguments are incompatible."""


class StatsPlotMissingImplementationError(Exception):
    pass


class UnsupportedColormapError(Exception):
    """Raises when colormap is not supported."""
