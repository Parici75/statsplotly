import logging
from enum import Enum
from functools import wraps
from typing import Dict, Callable, Sequence, Any, Tuple, Type, TypeVar

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import validator

from statsplot import constants
from statsplot.exceptions import (
    StatsPlotInvalidArgumentError,
    StatsPlotSpecificationError,
)
from statsplot.plot_specifiers.data import (
    BaseModel,
    DataDimension,
    RegressionType,
    TraceData,
    CentralTendencyType,
    HistogramNormType,
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


class _TraceSpecifier(BaseModel):
    @staticmethod
    def remove_nans(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(
            self: Type[TS], data: pd.Series, *args: Any, **kwargs: Any
        ) -> Callable:
            return function(self, data.dropna(), *args, **kwargs)

        return wrapper


class ScatterSpecifier(_TraceSpecifier):
    mode: TraceMode | None
    regression_type: RegressionType | None

    @validator("mode", pre=True)
    def check_mode(cls, value: str | None) -> TraceMode | None:
        try:
            return TraceMode(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, TraceMode  # type: ignore
            ) from exc

    @validator("regression_type", pre=True)
    def check_regression_type(cls, value: str | None) -> RegressionType | None:
        try:
            return RegressionType(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, RegressionType  # type: ignore
            ) from exc


class HistogramSpecifier(_TraceSpecifier):
    hist: bool | None
    cumulative: bool | None
    step: bool | None
    kde: bool | None
    rug: bool | None
    histnorm: HistogramNormType
    bin_edges: np.ndarray | None
    bins: int | str = 10
    central_tendency: CentralTendencyType | None
    data_type: np.dtype
    dimension: DataDimension

    @validator("cumulative")
    def check_cumulative(
        cls, value: bool | None, values: Dict[str, Any]
    ) -> bool | None:
        if value and not values.get("hist"):
            raise StatsPlotSpecificationError(
                "Cumulative histogram requires histogram bins plotting"
            )
        return value

    @validator("kde")
    def check_kde(
        cls, value: bool | None, values: Dict[str, Any]
    ) -> bool | None:
        if value and values.get("cumulative"):
            raise StatsPlotSpecificationError(
                "KDE is incompatible with cumulative histogram plotting"
            )
        if value and values.get("step"):
            raise StatsPlotSpecificationError(
                "KDE is incompatible with step histogram plotting"
            )
        return value

    @validator("bins")
    def check_bins(
        cls, value: str | Sequence | int | None
    ) -> str | Sequence | int:
        return (
            value
            if value is not None
            else constants.DEFAULT_HISTOGRAM_BIN_COMPUTATION_METHOD
        )

    @validator("histnorm", pre=True)
    def check_histnorm(
        cls, value: str | None, values: Dict[str, Any]
    ) -> HistogramNormType:
        if values.get("kde"):
            logger.warning(
                f"KDE plotting is on, turning histogram norm to {HistogramNormType.PROBABILITY_DENSITY}"
            )
            return HistogramNormType.PROBABILITY_DENSITY
        try:
            return (
                HistogramNormType(value)
                if value is not None
                else HistogramNormType.COUNT
            )
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, HistogramNormType  # type: ignore
            ) from exc

    @validator("central_tendency", pre=True)
    def check_central_tendency(
        cls, value: str | None
    ) -> CentralTendencyType | None:
        try:
            return CentralTendencyType(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, CentralTendencyType  # type: ignore
            ) from exc

    @validator("dimension")
    def check_dimension(
        cls, value: DataDimension, values: Dict[str, Any]
    ) -> DataDimension:
        if not is_numeric_dtype((dtype := values.get("data_type"))):
            raise StatsPlotSpecificationError(
                f"Distribution of {value} values of type: `{dtype}` can not be computed"
            )
        return value

    @property
    def density(self) -> bool:
        return (
            True
            if self.histnorm is HistogramNormType.PROBABILITY_DENSITY
            else False
        )

    @_TraceSpecifier.remove_nans
    def histogram_bin_edges(self, data: pd.Series) -> Tuple[np.ndarray, float]:
        bin_edges = np.histogram_bin_edges(
            data,
            bins=self.bin_edges  # type: ignore
            if self.bin_edges is not None
            else self.bins,
        )
        bin_size = np.round(
            bin_edges[1] - bin_edges[0], 6
        )  # Round to assure smooth binning by plotly

        return bin_edges, bin_size

    @_TraceSpecifier.remove_nans
    def histogram(
        self, data: pd.Series
    ) -> Tuple[pd.Series, np.ndarray, float]:
        bin_edges, bin_size = self.histogram_bin_edges(data)
        hist, bin_edges = np.histogram(
            data, bins=bin_edges, density=self.density
        )

        # Normalize if applicable
        if (
            self.histnorm is HistogramNormType.PROBABILITY
            or self.histnorm is HistogramNormType.PERCENT
        ):
            hist = hist / sum(hist)
            if self.histnorm is HistogramNormType.PERCENT:
                hist = hist * 100

        return (
            pd.Series(
                hist, name=self.histnorm if len(self.histnorm) > 0 else "count"
            ),
            bin_edges,
            bin_size,
        )


class JointplotSpecifier(_TraceSpecifier):
    plot_type: JointplotType
    marginal_plot: MarginalPlotDimension | None
    histogram_specifier: Dict[DataDimension, HistogramSpecifier] | None
    scatter_specifier: ScatterSpecifier

    @validator("plot_type", pre=True)
    def check_jointplot_type(cls, value: str) -> JointplotType:
        try:
            return JointplotType(value)
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, JointplotType  # type: ignore
            ) from exc

    @validator("marginal_plot", pre=True)
    def check_marginal_plot(
        cls, value: str, values: Dict[str, Any]
    ) -> MarginalPlotDimension | None:
        try:
            return MarginalPlotDimension(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, MarginalPlotDimension  # type: ignore
            ) from exc

    @validator("scatter_specifier")
    def check_scatter_specifier(
        cls, value: ScatterSpecifier, values: Dict[str, Any]
    ) -> ScatterSpecifier:
        if value.regression_type is not None and (
            plot_type := values.get("plot_type")
        ) not in (
            JointplotType.SCATTER,
            JointplotType.SCATTER_KDE,
        ):
            raise StatsPlotSpecificationError(
                f"{value.regression_type} regression can not be displayed on a {plot_type} plot"
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
    ) -> Tuple[pd.Series, Tuple, Tuple]:
        assert self.histogram_specifier is not None
        x, y = data.iloc[:, 0], data.iloc[:, 1]
        xbin_edges, xbin_size = self.histogram_specifier[
            DataDimension.X
        ].histogram_bin_edges(x)
        ybin_edges, ybin_size = self.histogram_specifier[
            DataDimension.X
        ].histogram_bin_edges(y)

        hist, _, _ = np.histogram2d(
            x,
            y,
            bins=[xbin_edges, ybin_edges],
            density=self.histogram_specifier[DataDimension.X].density,
        )

        # Normalize if applicable
        if (
            (histnorm := self.histogram_specifier[DataDimension.X].histnorm)
            is HistogramNormType.PROBABILITY
            or histnorm is HistogramNormType.PERCENT
        ):
            hist = hist / sum(hist)
            if histnorm is HistogramNormType.PERCENT:
                hist = hist * 100

        return (
            pd.Series(np.ravel(hist), name="hist"),
            (xbin_edges, ybin_edges),
            (xbin_size, ybin_size),
        )

    def compute_histmap(
        self, trace_data: TraceData
    ) -> Tuple[pd.Series, pd.Series, np.ndarray]:
        assert trace_data.x_values is not None
        assert trace_data.y_values is not None
        assert self.histogram_specifier is not None

        if self.plot_type is JointplotType.X_HISTMAP:
            anchor_values, histogram_data = (
                trace_data.x_values,
                trace_data.y_values,
            )
            histogram_specifier = self.histogram_specifier[
                DataDimension.Y
            ].copy()
        elif self.plot_type is JointplotType.Y_HISTMAP:
            anchor_values, histogram_data = (
                trace_data.y_values,
                trace_data.x_values,
            )
            histogram_specifier = self.histogram_specifier[
                DataDimension.X
            ].copy()

        # Get and set uniform bin edges along anchor values
        bin_edges, bin_size = histogram_specifier.histogram_bin_edges(
            histogram_data
        )
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
                name=histogram_specifier.histnorm
                if len(histogram_specifier.histnorm) > 0
                else "count",
            ),
            np.tile(bin_centers, hist.shape[0]),
        )
