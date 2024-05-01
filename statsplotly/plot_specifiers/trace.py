from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_numeric_dtype
from pydantic import ValidationInfo, field_validator, model_validator

from statsplotly import constants
from statsplotly._base import BaseModel
from statsplotly.exceptions import StatsPlotSpecificationError
from statsplotly.plot_specifiers.data import (
    CentralTendencyType,
    DataDimension,
    DataHandler,
    DataTypes,
    HistogramNormType,
    RegressionType,
    TraceData,
)

logger = logging.getLogger(__name__)


class TraceMode(str, Enum):
    MARKERS = "markers"
    LINES = "lines"
    MARKERS_LINES = "markers+lines"
    LINES_TEXT = "lines+text"


class CategoricalPlotType(str, Enum):
    STRIP = "stripplot"
    VIOLIN = "violinplot"
    BOX = "boxplot"


class MarginalPlotDimension(str, Enum):
    X = "x"
    Y = "y"
    ALL = "all"


class JointplotType(str, Enum):
    SCATTER = "scatter"
    KDE = "kde"
    SCATTER_KDE = "scatter+kde"
    X_HISTMAP = "x_histmap"
    Y_HISTMAP = "y_histmap"
    HISTOGRAM = "histogram"


F = TypeVar("F", bound=Callable[..., Any])


class _TraceSpecifier(BaseModel):
    @staticmethod
    def remove_nans(function: F) -> F:
        @wraps(function)
        def wrapper(self: _TraceSpecifier, data: pd.Series, *args: Any, **kwargs: Any) -> F:
            return function(self, data.dropna(), *args, **kwargs)

        return cast(F, wrapper)


class _XYTraceValidator(BaseModel):
    data_types: DataTypes

    @model_validator(mode="after")
    def validate_model(self) -> _XYTraceValidator:
        if not (self.data_types.x is not None and self.data_types.y is not None):
            raise StatsPlotSpecificationError(
                f"Both `x` and `y`dimensions must be supplied for {self.__class__.__name__}"
            )
        return self


class PlotOrientation(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class OrientedPlotSpecifier(BaseModel):
    prefered_orientation: PlotOrientation | None = None
    data_types: DataTypes

    @property
    def orientation(self) -> PlotOrientation:
        if self.data_types.x is None:
            return PlotOrientation.HORIZONTAL
        if self.data_types.y is None:
            return PlotOrientation.VERTICAL
        if self.prefered_orientation is not None:
            return self.prefered_orientation
        if is_numeric_dtype(self.data_types.x):
            return PlotOrientation.HORIZONTAL
        return PlotOrientation.VERTICAL

    @property
    def anchor_dimension(self) -> DataDimension:
        if self.orientation is PlotOrientation.VERTICAL:
            return DataDimension.X
        return DataDimension.Y

    @property
    def anchored_dimension(self) -> DataDimension | None:
        if self.data_types.x is None or self.data_types.y is None:
            return None
        if self.anchor_dimension is DataDimension.X:
            return DataDimension.Y
        return DataDimension.X


class ScatterSpecifier(_TraceSpecifier, _XYTraceValidator):
    mode: TraceMode | None = None
    regression_type: RegressionType | None = None


class CategoricalPlotSpecifier(OrientedPlotSpecifier, _TraceSpecifier, _XYTraceValidator):
    plot_type: CategoricalPlotType

    @field_validator("plot_type", mode="before")
    def validate_plot_type(cls, value: str | None) -> CategoricalPlotType:
        if value is None:
            return CategoricalPlotType.STRIP
        return CategoricalPlotType(value)

    @model_validator(mode="after")
    def validate_model(self) -> CategoricalPlotSpecifier:
        if self.data_types.color is not None and self.plot_type is not CategoricalPlotType.STRIP:
            raise StatsPlotSpecificationError(
                f"Only slice-level color data can be specified with `{self.plot_type.value}`, got marker-level argument `color={self.data_types.color}`"
            )
        return self

    def get_category_strip_map(
        self, data_handler: DataHandler
    ) -> dict[DataDimension, dict[str, Any]] | None:
        categorical_data = data_handler.get_data(self.anchor_dimension)
        if categorical_data is None:
            raise StatsPlotSpecificationError(
                f"Could not find `{self.anchor_dimension.value}` in data pointer"
            )
        if is_numeric_dtype(categorical_data):
            return None

        categorical_data_dict: dict[str, Any] = {}
        for i, x_level in enumerate(np.sort(categorical_data.dropna().astype(str).unique()), 1):
            categorical_data_dict[x_level] = i

        return {self.anchor_dimension: categorical_data_dict}


class HistogramSpecifier(_TraceSpecifier):
    hist: bool | None = None
    cumulative: bool | None = None
    step: bool | None = None
    ecdf: bool | None = None
    kde: bool | None = None
    rug: bool | None = None
    histnorm: HistogramNormType
    bin_edges: NDArray[Any] | None = None
    bins: str | list[float] | int
    central_tendency: CentralTendencyType | None = None
    data_type: np.dtype[Any]
    dimension: DataDimension

    @field_validator("cumulative")
    def check_cumulative(cls, value: bool | None, info: ValidationInfo) -> bool | None:
        if value and not info.data.get("hist"):
            raise StatsPlotSpecificationError(
                "Cumulative histogram requires histogram bins plotting"
            )
        return value

    @field_validator("bins", mode="before")
    def check_bins(cls, value: str | list[float] | int | None) -> str | list[float] | int:
        return value if value is not None else constants.DEFAULT_HISTOGRAM_BIN_COMPUTATION_METHOD

    @field_validator("histnorm", mode="before")
    def check_histnorm(cls, value: str | None, info: ValidationInfo) -> str | None:
        if info.data.get("kde"):
            if value is None:
                logger.info(
                    f"Setting histogram norm to {HistogramNormType.PROBABILITY_DENSITY.value} with"
                    " KDE plotting"
                )
                return HistogramNormType.PROBABILITY_DENSITY

        return value or HistogramNormType.COUNT

    @field_validator("dimension")
    def check_dimension(cls, value: DataDimension, info: ValidationInfo) -> DataDimension:
        if not is_numeric_dtype(dtype := info.data.get("data_type")):
            raise StatsPlotSpecificationError(
                f"Distribution of {value} values of type: `{dtype}` can not be computed"
            )
        return value

    @model_validator(mode="after")
    def check_parameter_consistency(self: HistogramSpecifier) -> HistogramSpecifier:
        if self.ecdf and self.histnorm is HistogramNormType.PROBABILITY_DENSITY:
            raise StatsPlotSpecificationError(
                "Histogram norm can not be set to"
                f" {HistogramNormType.PROBABILITY_DENSITY.value} with ECDF plotting"
            )

        if self.kde:
            if self.histnorm is not HistogramNormType.PROBABILITY_DENSITY:
                raise StatsPlotSpecificationError(
                    "Histogram norm must be set to"
                    f" {HistogramNormType.PROBABILITY_DENSITY.value} with KDE plotting,"
                    f" got `{self.histnorm.name}`"
                )

            if self.cumulative:
                raise StatsPlotSpecificationError(
                    "KDE is incompatible with cumulative histogram plotting"
                )

            if self.step:
                raise StatsPlotSpecificationError(
                    "KDE is incompatible with step histogram plotting"
                )

        return self

    @property
    def density(self) -> bool:
        return True if self.histnorm is HistogramNormType.PROBABILITY_DENSITY else False

    def get_distribution_max_value(self, data: pd.Series) -> float:
        dist_function: Callable[[Any], Any]
        if self.ecdf:
            dist_function = self.compute_ecdf
        else:
            dist_function = self.compute_histogram

        return dist_function(data)[0].max()

    @_TraceSpecifier.remove_nans
    def get_histogram_bin_edges(self, data: pd.Series) -> tuple[NDArray[Any], float]:
        bin_edges = np.histogram_bin_edges(
            data,
            bins=self.bin_edges if self.bin_edges is not None else self.bins,
        )
        bin_size = np.round(
            bin_edges[1] - bin_edges[0], 6
        )  # Round to assure smooth binning by plotly

        return bin_edges, bin_size

    @_TraceSpecifier.remove_nans
    def compute_histogram(self, data: pd.Series) -> tuple[pd.Series, NDArray[Any], float]:
        bin_edges, bin_size = self.get_histogram_bin_edges(data)
        hist, bin_edges = np.histogram(data, bins=bin_edges, density=self.density)

        # Normalize if applicable
        if (
            self.histnorm is HistogramNormType.PROBABILITY
            or self.histnorm is HistogramNormType.PERCENT
        ):
            hist = hist / sum(hist)
            if self.histnorm is HistogramNormType.PERCENT:
                hist = hist * 100

        if self.cumulative:
            hist = np.cumsum(hist)

        return (
            pd.Series(hist, name=self.histnorm if len(self.histnorm) > 0 else "count"),
            bin_edges,
            bin_size,
        )

    @_TraceSpecifier.remove_nans
    def compute_ecdf(self, data: pd.Series) -> tuple[pd.Series, NDArray[Any]]:
        unique_values, counts = np.unique(np.sort(data), return_counts=True)

        cdf = np.cumsum(counts)

        if (
            self.histnorm is HistogramNormType.PROBABILITY
            or self.histnorm is HistogramNormType.PERCENT
        ):
            cdf = cdf / data.size
            if self.histnorm is HistogramNormType.PERCENT:
                cdf = cdf * 100

        return (
            pd.Series(cdf, name=self.histnorm if len(self.histnorm) > 0 else "count"),
            unique_values,
        )


class JointplotSpecifier(_TraceSpecifier):
    plot_type: JointplotType
    marginal_plot: MarginalPlotDimension | None = None
    histogram_specifier: dict[DataDimension, HistogramSpecifier] | None = None
    scatter_specifier: ScatterSpecifier

    @field_validator("scatter_specifier")
    def check_scatter_specifier(
        cls, value: ScatterSpecifier, info: ValidationInfo
    ) -> ScatterSpecifier:
        if value.regression_type is not None and (plot_type := info.data.get("plot_type")) not in (
            JointplotType.SCATTER,
            JointplotType.SCATTER_KDE,
        ):
            raise StatsPlotSpecificationError(
                f"{value.regression_type.value} regression can not be displayed on a"
                f" {plot_type} plot"
            )
        return value

    @property
    def plot_kde(self) -> bool:
        return self.plot_type in (JointplotType.KDE, JointplotType.SCATTER_KDE)

    @property
    def plot_scatter(self) -> bool:
        return self.plot_type in (
            JointplotType.SCATTER,
            JointplotType.SCATTER_KDE,
        )

    @property
    def plot_x_distribution(self) -> bool:
        return self.marginal_plot in (
            MarginalPlotDimension.X,
            MarginalPlotDimension.ALL,
        )

    @property
    def plot_y_distribution(self) -> bool:
        return self.marginal_plot in (
            MarginalPlotDimension.Y,
            MarginalPlotDimension.ALL,
        )

    @_TraceSpecifier.remove_nans
    def histogram2d(
        self, data: pd.DataFrame
    ) -> tuple[pd.Series, tuple[NDArray[Any], NDArray[Any]], tuple[float, float]]:
        if self.histogram_specifier is None:
            raise ValueError("`histogram_specifier` can not be `None`")
        x, y = data.iloc[:, 0], data.iloc[:, 1]
        xbin_edges, xbin_size = self.histogram_specifier[DataDimension.X].get_histogram_bin_edges(x)
        ybin_edges, ybin_size = self.histogram_specifier[DataDimension.Y].get_histogram_bin_edges(y)

        hist, _, _ = np.histogram2d(
            x,
            y,
            bins=[xbin_edges, ybin_edges],
            density=self.histogram_specifier[DataDimension.X].density,
        )

        # Normalize if applicable
        if (
            histnorm := self.histogram_specifier[DataDimension.X].histnorm
        ) is HistogramNormType.PROBABILITY or histnorm is HistogramNormType.PERCENT:
            hist = hist / sum(hist)
            if histnorm is HistogramNormType.PERCENT:
                hist = hist * 100

        return (
            pd.Series(np.ravel(hist), name="hist"),
            (xbin_edges, ybin_edges),
            (xbin_size, ybin_size),
        )

    def compute_histmap(self, trace_data: TraceData) -> tuple[pd.Series, pd.Series, NDArray[Any]]:
        if (
            trace_data.x_values is None
            or trace_data.y_values is None
            or self.histogram_specifier is None
        ):
            raise ValueError("x_values, y_values and histogram_specifier can not be `None`")

        if self.plot_type is JointplotType.X_HISTMAP:
            anchor_values, histogram_data = (
                trace_data.x_values,
                trace_data.y_values,
            )
            histogram_specifier = self.histogram_specifier[DataDimension.Y].model_copy()
        elif self.plot_type is JointplotType.Y_HISTMAP:
            anchor_values, histogram_data = (
                trace_data.y_values,
                trace_data.x_values,
            )
            histogram_specifier = self.histogram_specifier[DataDimension.X].model_copy()

        # Get and set uniform bin edges along anchor values
        bin_edges, bin_size = histogram_specifier.get_histogram_bin_edges(histogram_data)
        histogram_specifier.bin_edges = bin_edges

        # Initialize histogram array
        hist = np.zeros((len(anchor_values.unique()), len(bin_edges) - 1))
        for i, anchor_value in enumerate(anchor_values.unique()):
            hist[i, :], _, _ = histogram_specifier.compute_histogram(
                histogram_data[anchor_values == anchor_value]
            )

        # Bin centers
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        return (
            pd.Series(
                np.repeat(anchor_values.unique(), hist.shape[1]),
                name=anchor_values.name,
            ),
            pd.Series(
                np.ravel(hist),
                name=(
                    histogram_specifier.histnorm
                    if len(histogram_specifier.histnorm) > 0
                    else "count"
                ),
            ),
            np.tile(bin_centers, hist.shape[0]),
        )
