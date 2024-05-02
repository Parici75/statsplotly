"""Custom exceptions."""


class StatsPlotSpecificationError(ValueError):
    """Raises when plot arguments are incompatibles."""


class StatsPlotMissingImplementationError(Exception):
    pass


class UnsupportedColormapError(Exception):
    """Raises when colormap is not supported."""
