from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import scipy as sc
from numpy.typing import NDArray
from pydantic import ValidationInfo, field_validator, model_validator

from statsplotly import constants
from statsplotly._base import BaseModel
from statsplotly.exceptions import (
    StatsPlotMissingImplementationError,
    StatsPlotSpecificationError,
)

from ._utils import rand_jitter
from .statistics import range_normalize, sem

logger = logging.getLogger(__name__)


class DataDimension(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"


class SliceTraceType(str, Enum):
    ALL_DATA = "all data"
    SLICE = "slice"


class NormalizationType(str, Enum):
    CENTER = "center"
    MIN_MAX = "minmax"
    ZSCORE = "zscore"


class RegressionType(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    INVERSE = "inverse"


class AggregationType(str, Enum):
    MEAN = "mean"
    GEO_MEAN = "geo_mean"
    COUNT = "count"
    MEDIAN = "median"
    PERCENT = "percent"
    PROBABILITY = "probability"
    SUM = "sum"


class CentralTendencyType(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"


class ErrorBarType(str, Enum):
    SEM = "sem"
    IQR = "iqr"
    STD = "std"
    GEO_STD = "geo_std"
    BOOTSTRAP = "bootstrap"


class HistogramNormType(str, Enum):
    COUNT = ""
    PERCENT = "percent"
    PROBABILITY = "probability"
    PROBABILITY_DENSITY = "probability density"


AGG_TO_ERROR_MAPPING: dict[CentralTendencyType, ErrorBarType] = {
    CentralTendencyType.MEAN: ErrorBarType.STD,
    CentralTendencyType.MEDIAN: ErrorBarType.IQR,
}


TRACE_DIMENSION_MAP = dict(
    zip(
        DataDimension,
        ["_".join((dimension.value, "values")) for dimension in DataDimension],
        strict=True,
    )
)

AGG_DIMENSION_TO_ERROR_DIMENSION = dict(
    zip(
        DataDimension,
        ["_".join(("error", dimension.value)) for dimension in DataDimension],
        strict=True,
    )
)


F = TypeVar("F", bound=Callable[..., Any])


class DataTypes(BaseModel):
    x: np.dtype[Any] | None = None
    y: np.dtype[Any] | None = None
    z: np.dtype[Any] | None = None
    color: np.dtype[Any] | None = None
    marker: np.dtype[Any] | None = None
    size: np.dtype[Any] | None = None
    text: np.dtype[Any] | None = None


class DataPointer(BaseModel):
    x: str | None = None
    y: str | None = None
    z: str | None = None
    slicer: str | None = None
    shaded_error: str | None = None
    error_x: str | None = None
    error_y: str | None = None
    error_z: str | None = None
    color: str | None = None
    marker: str | None = None
    opacity: str | float | None = None
    size: str | float | None = None
    text: str | None = None

    @model_validator(mode="after")
    def check_missing_dimension(self) -> DataPointer:
        if self.x is None and self.y is None:
            raise ValueError("Both `x` and `y` dimensions can not be None")
        return self

    @property
    def text_identifiers(self) -> list[str] | None:
        if self.text is not None:
            return self.text.split("+")
        return None


class DataHandler(BaseModel):
    data: pd.DataFrame
    data_pointer: DataPointer
    slice_order: list[Any] | None = None
    slice_logical_indices: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_pointers_in_data(self) -> DataHandler:
        for dimension in DataDimension:
            if (
                pointer := getattr(self.data_pointer, dimension)
            ) is not None and pointer not in self.data.columns:
                raise ValueError(f"{pointer} is not present in {self.data.columns}")

        return self

    @field_validator("data")
    def check_dataframe_format(cls, value: pd.DataFrame) -> pd.DataFrame:
        if len(value.columns.names) > 1:
            raise ValueError(
                "Multi-indexed columns are not supported, flatten the header before calling"
                " statsplotly"
            )
        return value

    @field_validator("data")
    def convert_categorical_dtype_columns(cls, value: pd.DataFrame) -> pd.DataFrame:
        for column in value.columns:
            if isinstance(value[column].dtype, pd.CategoricalDtype):
                logger.debug(f"Casting categorical {column} data to string")
                value[column] = value[column].astype(str)
        return value

    @property
    def slice_levels(self) -> list[str]:
        if self.slice_logical_indices is not None:
            return list(self.slice_logical_indices.keys())
        return []

    @property
    def n_slices(self) -> int:
        if self.slice_logical_indices is not None:
            return len(self.slice_logical_indices)
        return 1

    @property
    def data_types(self) -> DataTypes:
        dtypes = self.data.dtypes
        data_types: dict[str, Any] = {}
        for pointer, variable in self.data_pointer.model_dump().items():
            if variable in self.data.columns:
                data_types[pointer] = dtypes.loc[variable]

        return DataTypes.model_validate(data_types)

    @property
    def _slicer_groupby_data(self) -> pd.DataFrame | pd.Grouper:
        if self.data_pointer.slicer is not None:
            return self.data.groupby(self.data_pointer.slicer, sort=False)
        return self.data

    @staticmethod
    def _get_data_slice_indices(
        slice_ids: pd.Series, slice_order: list[str] | None
    ) -> dict[str, NDArray[Any]]:
        if slice_order is not None:
            if len(excluded_slices := set(slice_ids.unique()).difference(set(slice_order))) > 0:
                logger.info(
                    f"{list(excluded_slices)} slices are not present in slices {slice_order} and"
                    " will not be plotted"
                )
            slices = []
            for slice_id in slice_order:
                if slice_id not in slice_ids.to_numpy():
                    raise ValueError(
                        f"Invalid slice identifier: '{slice_id}' could not be found in"
                        f" '{slice_ids.name}'"
                    )
                slices.append(str(slice_id))
        else:
            slices = slice_ids.dropna().unique().astype(str)

        logical_indices: dict[str, NDArray[Any]] = {}
        for slice_id in slices:
            logical_indices[slice_id] = (slice_ids.astype(str) == slice_id).to_numpy()

        return logical_indices

    @staticmethod
    def to_dataframe(function: F) -> pd.DataFrame:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
            pandas_output = function(*args, **kwargs)
            if len(pandas_output.shape) == 1:
                return pandas_output.to_frame().transpose()
            return pandas_output

        return wrapper

    @classmethod
    def build_handler(
        cls,
        data: pd.DataFrame,
        data_pointer: DataPointer,
        slice_order: list[str] | None = None,
    ) -> DataHandler:
        slice_logical_indices = None

        data = data.reset_index()
        if data_pointer.slicer is not None:
            data = data.dropna(subset=data_pointer.slicer)
            slice_logical_indices = cls._get_data_slice_indices(
                slice_ids=data[data_pointer.slicer], slice_order=slice_order
            )

        return cls(
            data=data,
            data_pointer=data_pointer,
            slice_logical_indices=slice_logical_indices,
        )

    @to_dataframe
    def get_mean(self, dimension: str) -> pd.DataFrame:
        def std(x: pd.Series) -> list[float]:
            return [x.mean() - x.std(), x.mean() + x.std()]

        return self._slicer_groupby_data[getattr(self.data_pointer, dimension)].agg([np.mean, std])

    @to_dataframe
    def get_median(self, dimension: str) -> pd.DataFrame:
        def iqr(x: pd.Series) -> list[float]:
            return np.quantile(x, constants.IQR, axis=0).tolist()

        return self._slicer_groupby_data[getattr(self.data_pointer, dimension)].agg(
            [np.median, iqr]
        )

    def get_data(self, dimension: str) -> pd.Series | None:
        if (dimension_pointer := getattr(self.data_pointer, dimension)) is None:
            return None
        if dimension_pointer not in self.data.columns:
            return None
        return self.data[dimension_pointer]

    def iter_slices(
        self,
    ) -> Generator[tuple[str, pd.DataFrame], None, None]:
        levels: list[str] = self.slice_levels or (
            [self.data_pointer.y]
            if self.data_pointer.y is not None
            else [self.data_pointer.x or ""]
        )
        for level in levels:
            trace_data = (
                self.data.loc[self.slice_logical_indices[level]]
                if self.slice_logical_indices is not None
                else self.data
            )
            yield level, trace_data


class DataProcessor(BaseModel):
    data_values_map: dict[DataDimension, dict[str, Any]] | None = None
    jitter_settings: dict[DataDimension, float] | None = None
    normalizer: dict[DataDimension, NormalizationType] | None = None

    @field_validator("normalizer", mode="before")
    def check_normalizer(
        cls, value: dict[DataDimension, Any]
    ) -> dict[DataDimension, NormalizationType]:
        validated_norm: dict[DataDimension, NormalizationType] = {}
        for dimension, normalization in value.items():
            if normalization is not None:
                validated_norm.update({dimension: normalization})

        return validated_norm

    @staticmethod
    def jitter_data(data_series: pd.Series, jitter_amount: float) -> pd.Series:
        if jitter_amount == 0:
            return data_series

        return pd.Series(rand_jitter(data_series, jitter_amount), name=data_series.name)

    @staticmethod
    def normalize_data(data_series: pd.Series, normalizer: NormalizationType) -> pd.Series:
        match normalizer:
            case NormalizationType.CENTER:
                return data_series - data_series.mean()
            case NormalizationType.MIN_MAX:
                return pd.Series(
                    range_normalize(data_series.to_numpy(), 0, 1),
                    name=data_series.name,
                )
            case NormalizationType.ZSCORE:
                return pd.Series(
                    sc.stats.zscore(data_series.to_numpy(), nan_policy="omit"),
                    name=data_series.name,
                )

    def process_trace_data(self, trace_data: dict[str, pd.Series]) -> pd.Series:
        if self.data_values_map is not None:
            for dimension, values_map in self.data_values_map.items():
                trace_data[TRACE_DIMENSION_MAP[dimension]] = (
                    trace_data[TRACE_DIMENSION_MAP[dimension]]
                    .astype(str)
                    .map(lambda x, values_map=values_map: values_map[x])
                )

        if self.normalizer is not None:
            for dimension, normalizer in self.normalizer.items():
                if normalizer is None:
                    continue
                try:
                    trace_data[TRACE_DIMENSION_MAP[dimension]] = self.normalize_data(
                        data_series=trace_data[TRACE_DIMENSION_MAP[dimension]],
                        normalizer=normalizer,
                    )
                except TypeError:
                    logger.error(
                        f"Dimension {dimension.value} of type"
                        f" {trace_data[TRACE_DIMENSION_MAP[dimension]].dtype} can not be normalized"
                        f" with {normalizer.value}"
                    )

        if self.jitter_settings is not None:
            for dimension, jitter_amount in self.jitter_settings.items():
                try:
                    trace_data[TRACE_DIMENSION_MAP[dimension]] = self.jitter_data(
                        data_series=trace_data[TRACE_DIMENSION_MAP[dimension]],
                        jitter_amount=jitter_amount,
                    )
                except TypeError:
                    logger.error(
                        f"Dimension {dimension.value} of type"
                        f" {trace_data[TRACE_DIMENSION_MAP[dimension]].dtype} can not be jittered"
                    )

        return trace_data


class AggregationSpecifier(BaseModel):
    aggregation_func: AggregationType | Callable[[Any], float] | None = None
    aggregated_dimension: DataDimension
    error_bar: ErrorBarType | Callable[[Any], NDArray[Any]] | None = None
    data_types: DataTypes
    data_pointer: DataPointer

    @field_validator("error_bar")
    def check_error_bar(
        cls, value: ErrorBarType | None, info: ValidationInfo
    ) -> ErrorBarType | None:
        if value is not None and (
            (agg_func := info.data.get("aggregation_func")) is None
            or agg_func is AggregationType.COUNT
        ):
            raise StatsPlotSpecificationError(
                f"Plotting error bar requires one of "
                f"{[member.value for member in AggregationType if member is not AggregationType.COUNT]} "  # noqa: E501
                f"aggregation function"
            )
        return value

    @model_validator(mode="after")
    def check_aggregation_specifier(self) -> AggregationSpecifier:
        if (aggregation_func := self.aggregation_func) is not None:
            if getattr(self.data_pointer, self.aggregated_dimension) is None:
                raise StatsPlotSpecificationError(
                    f"aggregation dimension `{self.aggregated_dimension}` not found in the data"
                )

            # text can not be displayed along aggregation trace
            if self.data_pointer.text is not None:
                logger.warning("Text data can not be displayed along aggregated data")

            if aggregation_func is AggregationType.COUNT:
                if (
                    sum(
                        [
                            dimension is not None
                            for dimension in [self.data_pointer.x, self.data_pointer.y]
                        ]
                    )
                    > 1
                ):
                    raise StatsPlotSpecificationError(
                        f"{aggregation_func.value} aggregation only applies to one dimension"
                    )

        return self

    @property
    def reference_dimension(self) -> DataDimension:
        if self.aggregation_func is AggregationType.COUNT:
            return self.aggregated_dimension
        if self.aggregated_dimension is DataDimension.X:
            return DataDimension.Y
        return DataDimension.X

    @property
    def aggregation_plot_dimension(self) -> DataDimension:
        if self.aggregation_func is AggregationType.COUNT:
            return (
                DataDimension.Y if self.aggregated_dimension is DataDimension.X else DataDimension.X
            )
        return self.aggregated_dimension

    @property
    def reference_data(self) -> str | None:
        if self.aggregation_func is AggregationType.COUNT:
            return self.aggregated_data
        if self.reference_dimension is DataDimension.X:
            return self.data_pointer.x
        return self.data_pointer.y

    @property
    def aggregated_data(self) -> str:
        return getattr(self.data_pointer, self.aggregated_dimension)


class _BaseTraceData(BaseModel):
    x_values: pd.Series | None = None
    y_values: pd.Series | None = None
    z_values: pd.Series | None = None
    shaded_error: pd.Series | None = None
    error_x: pd.Series | None = None
    error_y: pd.Series | None = None
    error_z: pd.Series | None = None
    text_data: str | pd.Series | None = None
    color_data: str | pd.Series | None = None
    marker_data: str | pd.Series | None = None
    size_data: float | pd.Series | None = None
    opacity_data: float | pd.Series | None = None

    @field_validator("error_x", "error_y", "error_z")
    def check_error_data(cls, value: pd.Series | None) -> pd.Series | None:
        if value is None:
            return value

        if not all(
            value.apply(
                lambda x: np.issubdtype(np.asarray(x).dtype, np.number)
                or any(xx is None for xx in x)
            )
        ):
            raise ValueError(f"{value.name} error data must be numeric")

        if not all(value.apply(lambda x: len(x) == 2)):  # noqa: PLR2004
            raise ValueError(
                f"{value.name} error data must be bidirectional to be plotted relative to the"
                " underlying data"
            )

        return value

    @classmethod
    def assemble_hover_text(
        cls, data: pd.DataFrame, text_pointers: list[str] | None
    ) -> pd.Series | None:
        """Converts text columns of a DataFrame into plotly text box"""
        if text_pointers is None:
            return None
        lines = []
        for col in text_pointers:
            lines.append(data[col].map(lambda x: str(col) + ": " + str(x)).tolist())  # noqa: B023

        return pd.Series(map("<br>".join, zip(*lines, strict=True)), name="hover_text")

    @classmethod
    def _build_trace_data_from_pointer(
        cls, data: pd.DataFrame, pointer: DataPointer
    ) -> dict[str, Any]:
        trace_data: dict[str, Any] = {}
        trace_data["x_values"] = data[pointer.x] if pointer.x is not None else None
        trace_data["y_values"] = data[pointer.y] if pointer.y is not None else None
        trace_data["z_values"] = data[pointer.z] if pointer.z is not None else None

        # errors
        trace_data["shaded_error"] = (
            data[pointer.shaded_error]
            if pointer.shaded_error in data.columns
            else pointer.shaded_error
        )
        trace_data["error_x"] = (
            data[pointer.error_x] if pointer.error_x in data.columns else pointer.error_x
        )
        trace_data["error_y"] = (
            data[pointer.error_y] if pointer.error_y in data.columns else pointer.error_y
        )
        trace_data["error_z"] = (
            data[pointer.error_z] if pointer.error_z in data.columns else pointer.error_z
        )

        trace_data["marker_data"] = (
            data[pointer.marker] if pointer.marker in data.columns else pointer.marker
        )
        trace_data["text_data"] = cls.assemble_hover_text(
            data=data, text_pointers=pointer.text_identifiers
        )
        trace_data["color_data"] = (
            data[pointer.color] if pointer.color in data.columns else pointer.color
        )
        trace_data["size_data"] = (
            range_normalize(
                data[pointer.size], constants.MIN_MARKER_SIZE, constants.MAX_MARKER_SIZE
            )
            if pointer.size in data.columns
            else pointer.size
        )
        trace_data["opacity_data"] = (
            range_normalize(data[pointer.opacity], 0, 1)
            if pointer.opacity in data.columns
            else pointer.opacity
        )

        return trace_data


class TraceData(_BaseTraceData):
    @classmethod
    def build_trace_data(
        cls,
        data: pd.DataFrame,
        pointer: DataPointer,
        processor: DataProcessor | None = None,
    ) -> TraceData:
        trace_data = cls._build_trace_data_from_pointer(data, pointer)
        if processor is not None:
            trace_data = processor.process_trace_data(trace_data)

        return cls.model_validate(trace_data)


class AggregationTraceData(TraceData):

    @classmethod
    def _compute_error_bar(
        cls,
        data_group: pd.Grouper,
        agg_function: F,
        error_bar: ErrorBarType | Callable[[Any], NDArray[Any]],
    ) -> pd.Series:
        if error_bar in (ErrorBarType.STD, ErrorBarType.SEM):
            data_agg = data_group.apply(agg_function)
            match error_bar:
                case ErrorBarType.STD:
                    error_data = data_group.std()

                case ErrorBarType.SEM:
                    error_data = data_group.apply(
                        lambda series: sem(series, 1 - constants.CI_ALPHA)
                    )

            return pd.Series(
                zip(data_agg - error_data, data_agg + error_data, strict=True),
                name=data_agg.name,
            )

        if error_bar is ErrorBarType.GEO_STD:
            data_agg = data_group.apply(agg_function)
            error_data = data_group.apply(lambda series: np.exp(np.std(np.log(series))))
            return pd.Series(
                zip(data_agg / error_data, data_agg * error_data, strict=True),
                name=data_agg.name,
            )

        if error_bar is ErrorBarType.IQR:
            return data_group.apply(lambda series: np.quantile(series, constants.IQR))

        if error_bar is ErrorBarType.BOOTSTRAP:
            # bootstrap accepts a sequence of data
            return data_group.apply(
                lambda x: np.array(
                    sc.stats.bootstrap(
                        (x,),
                        agg_function,
                        confidence_level=1 - constants.CI_ALPHA,
                    ).confidence_interval
                )
            )

        if isinstance(error_bar, Callable):  # type: ignore
            return data_group.apply(error_bar)

        raise StatsPlotMissingImplementationError(f"Unsupported error bar type: {error_bar}")

    @classmethod
    def _build_aggregation_data_from_pointer(
        cls,
        data: pd.DataFrame,
        aggregation_specifier: AggregationSpecifier,
    ) -> dict[str, Any]:
        trace_data: dict[str, Any] = {}

        trace_data[TRACE_DIMENSION_MAP[aggregation_specifier.reference_dimension]] = pd.Series(
            data[aggregation_specifier.reference_data].unique(),
            name=aggregation_specifier.reference_data,
        )
        if (
            aggregation_specifier.aggregation_func is AggregationType.COUNT
            or aggregation_specifier.aggregation_func is AggregationType.PROBABILITY
            or aggregation_specifier.aggregation_func is AggregationType.PERCENT
        ):
            _aggregated_values: list[NDArray[Any]] = []
            for reference_value in trace_data[
                TRACE_DIMENSION_MAP[aggregation_specifier.reference_dimension]
            ]:
                aggregated_value = (
                    data[aggregation_specifier.reference_data] == reference_value
                ).sum()
                if aggregation_specifier.aggregation_func is AggregationType.PROBABILITY:
                    _aggregated_values.append(
                        aggregated_value
                        / data[aggregation_specifier.reference_data].notnull().sum()
                    )
                elif aggregation_specifier.aggregation_func is AggregationType.PERCENT:
                    _aggregated_values.append(
                        aggregated_value
                        / data[aggregation_specifier.reference_data].notnull().sum()
                        * 100
                    )
                else:
                    _aggregated_values.append(aggregated_value)
            trace_data[TRACE_DIMENSION_MAP[aggregation_specifier.aggregation_plot_dimension]] = (
                pd.Series(
                    _aggregated_values,
                    name="_".join(
                        (
                            aggregation_specifier.aggregated_dimension,
                            aggregation_specifier.aggregation_func.value,
                        )
                    ),
                )
            )

        else:
            if isinstance(aggregation_specifier.aggregation_func, AggregationType):
                agg_func: Callable[[Any], Any]
                match aggregation_specifier.aggregation_func:
                    case AggregationType.MEAN:
                        agg_func = np.mean
                    case AggregationType.GEO_MEAN:
                        agg_func = sc.stats.mstats.gmean
                    case AggregationType.MEDIAN:
                        agg_func = np.median
                    case AggregationType.SUM:
                        agg_func = np.sum
            else:
                agg_func = aggregation_specifier.aggregation_func  # type: ignore

            trace_data[TRACE_DIMENSION_MAP[aggregation_specifier.aggregated_dimension]] = (
                data.groupby(aggregation_specifier.reference_data, sort=False)[
                    aggregation_specifier.aggregated_data
                ].apply(agg_func)
            )

            if aggregation_specifier.error_bar is not None:
                trace_data[
                    AGG_DIMENSION_TO_ERROR_DIMENSION[aggregation_specifier.aggregated_dimension]
                ] = cls._compute_error_bar(
                    data.groupby(aggregation_specifier.reference_data, sort=False)[
                        aggregation_specifier.aggregated_data
                    ],
                    agg_func,
                    aggregation_specifier.error_bar,
                )

        return trace_data

    @classmethod
    def build_aggregation_trace_data(
        cls, data: pd.DataFrame, aggregation_specifier: AggregationSpecifier
    ) -> AggregationTraceData:
        trace_data = cls._build_aggregation_data_from_pointer(data, aggregation_specifier)

        return cls.model_validate(trace_data)
