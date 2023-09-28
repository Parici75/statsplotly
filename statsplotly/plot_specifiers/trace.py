import logging
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_numeric_dtype
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from statsplotly import constants
from statsplotly.exceptions import (
    StatsPlotInvalidArgumentError,
    StatsPlotSpecificationError,
)
from statsplotly.plot_specifiers.data import (
    BaseModel,
    CentralTendencyType,
    DataDimension,
    HistogramNormType,
    RegressionType,
    TraceData,
)

logger = logging.getLogger(__name__)


class TraceMode(str, Enum):
    MARKERS = "markers"
    LINES = "lines"
    MARKERS_LINES = "markers+lines"


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


TS = TypeVar("TS", bound="_TraceSpecifier")
F = TypeVar("F", bound=Callable[..., Any])


class _TraceSpecifier(BaseModel):
    @staticmethod
    def remove_nans(function: F) -> F:
        @wraps(function)
        def wrapper(self: type[TS], data: pd.Series, *args: Any, **kwargs: Any) -> F:
            return function(self, data.dropna(), *args, **kwargs)

        return cast(F, wrapper)


class ScatterSpecifier(_TraceSpecifier):
    mode: TraceMode | None = None
    regression_type: RegressionType | None = None

    @field_validator("mode", mode="before")
    def check_mode(cls, value: str | None) -> TraceMode | None:
        try:
            return TraceMode(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(value, TraceMode) from exc  # type: ignore

    @field_validator("regression_type", mode="before")
    def check_regression_type(cls, value: str | None) -> RegressionType | None:
        try:
            return RegressionType(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(value, RegressionType) from exc  # type: ignore


class HistogramSpecifier(_TraceSpecifier):
    hist: bool | None = None
    cumulative: bool | None = None
    step: bool | None = None
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

    @field_validator("kde")
    def check_kde(cls, value: bool | None, info: ValidationInfo) -> bool | None:
        if value and info.data.get("cumulative"):
            raise StatsPlotSpecificationError(
                "KDE is incompatible with cumulative histogram plotting"
            )
        if value and info.data.get("step"):
            raise StatsPlotSpecificationError("KDE is incompatible with step histogram plotting")
        return value

    @field_validator("bins")
    def check_bins(cls, value: str | list[float] | int | None) -> str | list[float] | int:
        return value if value is not None else constants.DEFAULT_HISTOGRAM_BIN_COMPUTATION_METHOD

    @field_validator("histnorm", mode="before")
    def check_histnorm(cls, value: str | None, info: ValidationInfo) -> HistogramNormType:
        if info.data.get("kde"):
            if value is None:
                logger.info(
                    f"Setting histogram norm to {HistogramNormType.PROBABILITY_DENSITY.value} with"
                    " KDE plotting"
                )
                return HistogramNormType.PROBABILITY_DENSITY

            if value is not HistogramNormType.PROBABILITY_DENSITY:
                raise StatsPlotSpecificationError(
                    "Histogram norm must be set to"
                    f" {HistogramNormType.PROBABILITY_DENSITY.value} with KDE plotting, got {value}"
                )
        try:
            return HistogramNormType(value) if value is not None else HistogramNormType.COUNT
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(value, HistogramNormType) from exc  # type: ignore

    @field_validator("central_tendency")
    def check_central_tendency(cls, value: str | None) -> CentralTendencyType | None:
        try:
            return CentralTendencyType(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(value, CentralTendencyType) from exc  # type: ignore

    @field_validator("dimension")
    def check_dimension(cls, value: DataDimension, info: ValidationInfo) -> DataDimension:
        if not is_numeric_dtype(dtype := info.data.get("data_type")):
            raise StatsPlotSpecificationError(
                f"Distribution of {value} values of type: `{dtype}` can not be computed"
            )
        return value

    @property
    def density(self) -> bool:
        return True if self.histnorm is HistogramNormType.PROBABILITY_DENSITY else False

    @_TraceSpecifier.remove_nans
    def histogram_bin_edges(self, data: pd.Series) -> tuple[NDArray[Any], float]:
        bin_edges = np.histogram_bin_edges(
            data,
            bins=self.bin_edges if self.bin_edges is not None else self.bins,
        )
        bin_size = np.round(
            bin_edges[1] - bin_edges[0], 6
        )  # Round to assure smooth binning by plotly

        return bin_edges, bin_size

    @_TraceSpecifier.remove_nans
    def histogram(self, data: pd.Series) -> tuple[pd.Series, NDArray[Any], float]:
        bin_edges, bin_size = self.histogram_bin_edges(data)
        hist, bin_edges = np.histogram(data, bins=bin_edges, density=self.density)

        # Normalize if applicable
        if (
            self.histnorm is HistogramNormType.PROBABILITY
            or self.histnorm is HistogramNormType.PERCENT
        ):
            hist = hist / sum(hist)
            if self.histnorm is HistogramNormType.PERCENT:
                hist = hist * 100

        return (
            pd.Series(hist, name=self.histnorm if len(self.histnorm) > 0 else "count"),
            bin_edges,
            bin_size,
        )


class JointplotSpecifier(_TraceSpecifier):
    plot_type: JointplotType
    marginal_plot: MarginalPlotDimension | None = None
    histogram_specifier: dict[DataDimension, HistogramSpecifier] | None = None
    scatter_specifier: ScatterSpecifier

    @field_validator("plot_type", mode="before")
    def check_jointplot_type(cls, value: str) -> JointplotType:
        try:
            return JointplotType(value)
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(value, JointplotType) from exc  # type: ignore

    @field_validator("marginal_plot", mode="before")
    def check_marginal_plot(cls, value: str) -> MarginalPlotDimension | None:
        try:
            return MarginalPlotDimension(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, MarginalPlotDimension  # type: ignore
            ) from exc

    @field_validator("scatter_specifier")
    def check_scatter_specifier(
        cls, value: ScatterSpecifier, info: ValidationInfo
    ) -> ScatterSpecifier:
        if value.regression_type is not None and (plot_type := info.data["plot_type"]) not in (
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
        xbin_edges, xbin_size = self.histogram_specifier[DataDimension.X].histogram_bin_edges(x)
        ybin_edges, ybin_size = self.histogram_specifier[DataDimension.X].histogram_bin_edges(y)

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
        bin_edges, bin_size = histogram_specifier.histogram_bin_edges(histogram_data)
        histogram_specifier.bin_edges = bin_edges

        # Initialize histogram array
        hist = np.zeros((len(anchor_values.unique()), len(bin_edges) - 1))
        for i, anchor_value in enumerate(anchor_values.unique()):
            hist[i, :], _, _ = histogram_specifier.histogram(
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
