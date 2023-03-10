import logging
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Generator, Tuple

import numpy as np
import pandas as pd
import scipy as sc
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from pydantic import BaseModel as PydanticBaseModel
from pydantic import validator, root_validator

from pymodules.pandas_utils import unique_non_null
from statsplot import constants
from statsplot.exceptions import (
    StatsPlotInvalidArgumentError,
    StatsPlotMissingImplementationError,
    StatsPlotSpecificationError,
)
from statsplot.utils.colors_utils import rand_jitter
from statsplot.utils.stats_utils import range_normalize, sem

logger = logging.getLogger(__name__)


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class SliceTraceType(str, Enum):
    ALL_DATA = "all data"


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
    BOOTSTRAP = "bootstrap"


class HistogramNormType(str, Enum):
    COUNT = ""
    PERCENT = "percent"
    PROBABILITY = "probability"
    PROBABILITY_DENSITY = "probability density"


AGG_TO_ERROR_MAPPING: Dict[CentralTendencyType, ErrorBarType] = {
    CentralTendencyType.MEAN: ErrorBarType.STD,
    CentralTendencyType.MEDIAN: ErrorBarType.IQR,
}


class DataTypes(BaseModel):
    x: np.dtype | None
    y: np.dtype | None
    z: np.dtype | None
    color: np.dtype | None
    marker: np.dtype | None
    size: np.dtype | None
    text: np.dtype | None


class DataDimension(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"


TRACE_DIMENSION_MAP = dict(
    zip(
        DataDimension,
        ["_".join((dimension.value, "values")) for dimension in DataDimension],
    )
)


class DataPointer(BaseModel):
    x: str | None
    y: str | None
    z: str | None
    slicer: str | None
    shaded_error: str | None
    error_x: str | None
    error_y: str | None
    error_z: str | None
    color: str | float | None
    marker: str | None
    size: str | float | None
    text: str | None

    @validator("y", always=True)
    def check_missing_dimension(
        cls, value: str, values: Dict[str, Any]
    ) -> str | None:
        if value is None and values.get("x") is None:
            raise ValueError("Both x and y dimensions can not be None")
        return value

    @property
    def text_identifiers(self) -> List | None:
        if self.text is not None:
            return self.text.split("+")
        return None


class DataHandler(BaseModel):
    data: pd.DataFrame
    data_pointer: DataPointer
    slice_order: List[Any] | None
    slice_logical_indices: Dict[str, Any] | None

    @root_validator(pre=True)
    def check_pointers_in_data(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        data_pointer = values.get("data_pointer")
        data = values.get("data")
        assert isinstance(data, pd.DataFrame)

        for dimension in DataDimension:
            if (
                pointer := getattr(data_pointer, dimension)
            ) is not None and pointer not in data.columns:
                raise ValueError(f"{pointer} is not present in {data.columns}")
        values.update({"data": data})
        return values

    @validator("data")
    def check_dataframe_format(cls, value: pd.DataFrame) -> pd.DataFrame:
        if len(value.columns.names) > 1:
            raise ValueError(
                "Multi-indexed columns are not supported, flatten the header before calling statsplot"
            )
        return value

    @validator("data")
    def convert_categorical_dtype_columns(
        cls, value: pd.DataFrame
    ) -> pd.DataFrame:
        for column in value.columns:
            if is_categorical_dtype(value[column]):
                logger.debug(f"Casting categorical {column} data to string")
                value.loc[:, column] = value.loc[:, column].astype(str)
        return value

    @property
    def slice_levels(self) -> List[str]:
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
        data_types: Dict[str, Any] = {}
        for pointer, variable in self.data_pointer.dict().items():
            if variable in self.data.columns:
                data_types[pointer] = dtypes.loc[variable]

        return DataTypes.parse_obj(data_types)

    @property
    def slicer_groupby_data(self) -> pd.DataFrame | pd.Grouper:
        if self.data_pointer.slicer is not None:
            return self.data.groupby(self.data_pointer.slicer)
        return self.data

    @staticmethod
    def _get_data_slice_indices(
        slice_ids: pd.Series, slice_order: List[str] | None
    ) -> Dict[str, np.ndarray]:
        if slice_order is not None:
            if (
                len(
                    excluded_slices := set(slice_ids.unique()).difference(
                        set(slice_order)
                    )
                )
                > 0
            ):
                logger.info(
                    f"{list(excluded_slices)} slices are not present in slices {slice_order} and will not be plotted"
                )
            slices = []
            for slice_id in slice_order:
                if slice_id not in slice_ids.values:
                    raise ValueError(
                        f"Invalid slice identifier: '{slice_id}' could not be found in '{slice_ids.name}'"
                    )
                slices.append(str(slice_id))
        else:
            slices = unique_non_null(slice_ids).astype(str)

        logical_indices: Dict[str, np.ndarray] = {}
        for slice_id in slices:
            logical_indices[slice_id] = (
                slice_ids.astype(str) == slice_id
            ).values

        return logical_indices

    @staticmethod
    def to_dataframe(function: Callable) -> pd.DataFrame:
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
        slice_order: List[str] | None = None,
    ) -> "DataHandler":
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
        def std(x: pd.Series) -> List[float]:
            return [x.mean() - x.std(), x.mean() + x.std()]

        return self.slicer_groupby_data[
            getattr(self.data_pointer, dimension)
        ].agg([np.mean, std])

    @to_dataframe
    def get_median(self, dimension: str) -> pd.DataFrame:
        def iqr(x: pd.Series) -> List[float]:
            return np.quantile(x, constants.IQR, axis=0).tolist()

        return self.slicer_groupby_data[
            getattr(self.data_pointer, dimension)
        ].agg([np.median, iqr])

    def get_data(self, dimension: str) -> pd.Series | None:
        if (
            dimension_pointer := getattr(self.data_pointer, dimension)
        ) is None:
            return None
        if dimension_pointer not in self.data.columns:
            return None
        return self.data[dimension_pointer]

    def traces_data(
        self,
    ) -> Generator[Tuple[str, pd.DataFrame], None, None]:
        levels: List[str] = self.slice_levels or [""]
        for level in levels:
            trace_data = (
                self.data.loc[self.slice_logical_indices[level]]
                if self.slice_logical_indices is not None
                else self.data
            )
            yield level, trace_data


class DataProcessor(BaseModel):
    x_values_mapping: Dict[str, Any] | None
    jitter_settings: Dict[DataDimension, float] | None
    normalizer: Dict[DataDimension, NormalizationType] | None

    @validator("normalizer", pre=True)
    def check_normalizer(
        cls, value: Dict[DataDimension, Any]
    ) -> Dict[DataDimension, NormalizationType]:
        validated_norm: Dict[DataDimension, NormalizationType] = {}
        for dimension, normalization in value.items():
            if normalization is not None:
                try:
                    validated_norm.update(
                        {dimension: NormalizationType(normalization)}
                    )
                except ValueError as exc:
                    raise StatsPlotInvalidArgumentError(
                        normalization, NormalizationType  # type: ignore
                    ) from exc

        return validated_norm

    @staticmethod
    def jitter_data(data_series: pd.Series, jitter_amount: float) -> pd.Series:
        if jitter_amount == 0:
            return data_series
        jittered_data = pd.Series(
            rand_jitter(data_series, jitter_amount), name=data_series.name
        )
        return jittered_data

    @staticmethod
    def normalize_data(
        data_series: pd.Series, normalizer: NormalizationType
    ) -> pd.Series:
        match normalizer:
            case NormalizationType.CENTER:
                return data_series - data_series.mean()
            case NormalizationType.MIN_MAX:
                return pd.Series(
                    range_normalize(data_series.values, 0, 1),
                    name=data_series.name,
                )
            case NormalizationType.ZSCORE:
                return pd.Series(
                    sc.stats.zscore(data_series.values, nan_policy="omit"),
                    name=data_series.name,
                )

    def process_trace_data(
        self, trace_data: Dict[str, pd.Series]
    ) -> pd.Series:
        if self.x_values_mapping is not None:
            trace_data[TRACE_DIMENSION_MAP[DataDimension.X]] = trace_data[
                TRACE_DIMENSION_MAP[DataDimension.X]
            ].map(lambda x: self.x_values_mapping[x])

        if self.normalizer is not None:
            for dimension, normalizer in self.normalizer.items():
                if normalizer is None:
                    continue
                try:
                    trace_data[
                        TRACE_DIMENSION_MAP[dimension]
                    ] = self.normalize_data(
                        data_series=trace_data[TRACE_DIMENSION_MAP[dimension]],
                        normalizer=normalizer,
                    )
                except TypeError:
                    logger.error(
                        f"Dimension {dimension} of type {trace_data[TRACE_DIMENSION_MAP[dimension]].dtype} can not be normalized with {normalizer}"
                    )

        if self.jitter_settings is not None:
            for dimension, jitter_amount in self.jitter_settings.items():
                try:
                    trace_data[
                        TRACE_DIMENSION_MAP[dimension]
                    ] = self.jitter_data(
                        data_series=trace_data[TRACE_DIMENSION_MAP[dimension]],
                        jitter_amount=jitter_amount,
                    )
                except TypeError:
                    logger.error(
                        f"Dimension {dimension} of type {trace_data[TRACE_DIMENSION_MAP[dimension]].dtype} can not be jittered"
                    )

        return trace_data


class AggregationSpecifier(BaseModel):
    aggregation_func: AggregationType | None
    error_bar: ErrorBarType | None
    data_types: DataTypes
    data_pointer: DataPointer

    @validator("aggregation_func", pre=True)
    def check_aggregation_func(
        cls, value: str | None
    ) -> AggregationType | None:
        try:
            return AggregationType(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, AggregationType  # type: ignore
            ) from exc

    @validator("error_bar", pre=True)
    def check_error_bar(
        cls, value: str | None, values: Dict[str, Any]
    ) -> ErrorBarType | None:
        if value is not None and (
            (agg_func := values.get("aggregation_func")) is None
            or agg_func is AggregationType.COUNT
        ):
            raise StatsPlotSpecificationError(
                f"Error bar requires one of {[member.value for member in AggregationType if member is not AggregationType.COUNT]} aggregation function"
            )
        try:
            return ErrorBarType(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, ErrorBarType  # type: ignore
            ) from exc

    @validator("data_pointer")
    def check_data_pointer(
        cls, value: DataPointer, values: Dict[str, Any]
    ) -> DataPointer:
        # x dimension
        if value.x is None:
            value.x = "index"

        # y dimension
        if (
            value.y is None
            and (aggregation_func := values.get("aggregation_func"))
            is not AggregationType.COUNT
        ):
            raise StatsPlotSpecificationError(
                f"{aggregation_func} aggregation requires y data"
            )
        if value.y is not None:
            if (
                aggregation_func := values.get("aggregation_func")
            ) is AggregationType.COUNT:
                raise StatsPlotSpecificationError(
                    f"{aggregation_func} aggregation does not apply to y data"
                )

            assert (dtypes := values.get("data_types")) is not None
            if not is_numeric_dtype((y_dtype := dtypes.y)):
                raise StatsPlotSpecificationError(
                    f"{aggregation_func} aggregation requires numeric type y data, got: `{y_dtype}`"
                )

        if (
            value.text is not None
            and values.get("aggregation_func") is not None
        ):
            logger.warning(
                f"Text data can not be displayed along aggregated data"
            )
        return value


class _BaseTraceData(BaseModel):
    x_values: pd.Series | None
    y_values: pd.Series | None
    z_values: pd.Series | None
    shaded_error: pd.Series | None
    error_x: pd.Series | None
    error_y: pd.Series | None
    error_z: pd.Series | None
    text_data: str | pd.Series | None
    color_data: str | pd.Series | None
    marker_data: str | pd.Series | None
    size_data: float | pd.Series | None

    @validator("error_x", "error_y", "error_z")
    def check_error_data(cls, value: pd.Series | None) -> pd.Series | None:
        if value is None:
            return value

        if not all(
            value.apply(
                lambda x: np.issubdtype(np.asarray(x).dtype, np.number)
            )
        ):
            raise ValueError(f"{value.name} error data must be numeric")

        if not all(value.apply(lambda x: len(x) == 2)):
            raise ValueError(
                f"{value.name} error data must be bidirectional to be plotted relative to the underlying data"
            )

        return value

    @classmethod
    def assemble_hover_text(
        cls, data: pd.DataFrame, text_pointers: List | None
    ) -> pd.Series | None:
        """Converts text columns of a DataFrame into plotly text box"""
        if text_pointers is None:
            return None
        lines = []
        for col in text_pointers:
            series = data[col]
            lines.append(
                series.map(lambda x: str(series.name) + ": " + str(x)).tolist()
            )

        return pd.Series(map("<br>".join, zip(*lines)), name="hover_text")

    @classmethod
    def _build_trace_data_from_pointer(
        cls, data: pd.DataFrame, pointer: DataPointer
    ) -> Dict[str, Any]:
        trace_data: Dict[str, Any] = {}
        trace_data["x_values"] = (
            data[pointer.x] if pointer.x is not None else None
        )
        trace_data["y_values"] = (
            data[pointer.y] if pointer.y is not None else None
        )
        trace_data["z_values"] = (
            data[pointer.z] if pointer.z is not None else None
        )

        # errors
        trace_data["shaded_error"] = (
            data[pointer.shaded_error]
            if pointer.shaded_error in data.columns
            else pointer.shaded_error
        )
        trace_data["error_x"] = (
            data[pointer.error_x]
            if pointer.error_x in data.columns
            else pointer.error_x
        )
        trace_data["error_y"] = (
            data[pointer.error_y]
            if pointer.error_y in data.columns
            else pointer.error_y
        )
        trace_data["error_z"] = (
            data[pointer.error_z]
            if pointer.error_z in data.columns
            else pointer.error_z
        )

        trace_data["marker_data"] = (
            data[pointer.marker]
            if pointer.marker in data.columns
            else pointer.marker
        )
        trace_data["text_data"] = cls.assemble_hover_text(
            data=data, text_pointers=pointer.text_identifiers
        )
        trace_data["color_data"] = (
            data[pointer.color]
            if pointer.color in data.columns
            else pointer.color
        )
        trace_data["size_data"] = (
            data[[pointer.size]]
            .apply(
                lambda x: range_normalize(
                    x, constants.MIN_MARKER_SIZE, constants.MAX_MARKER_SIZE
                )
            )
            .squeeze()
            if pointer.size in data.columns
            else pointer.size
        )

        return trace_data


class TraceData(_BaseTraceData):
    @classmethod
    def build_trace_data(
        cls,
        data: pd.DataFrame,
        pointer: DataPointer,
        processor: DataProcessor | None = None,
    ) -> "TraceData":
        trace_data = cls._build_trace_data_from_pointer(data, pointer)
        if processor is not None:
            trace_data = processor.process_trace_data(trace_data)

        return cls.parse_obj(trace_data)


class AggregationTraceData(_BaseTraceData):
    @classmethod
    def _compute_error_bar(
        cls,
        data_group: pd.Grouper,
        agg_function: Callable,
        error_bar: ErrorBarType,
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
                zip(
                    data_agg - error_data,
                    data_agg + error_data,
                ),
                name=data_agg.name,
            )

        if error_bar is ErrorBarType.IQR:
            return data_group.apply(
                lambda series: np.quantile(series, constants.IQR)
            )

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

        raise StatsPlotMissingImplementationError(
            f"Unsupported error bar type: {error_bar}"
        )

    @classmethod
    def _build_aggregation_data_from_pointer(
        cls,
        data: pd.DataFrame,
        aggregation_specifier: AggregationSpecifier,
    ) -> Dict[str, Any]:
        trace_data: Dict[str, Any] = {}
        assert (x := aggregation_specifier.data_pointer.x) is not None
        y = aggregation_specifier.data_pointer.y

        trace_data["x_values"] = pd.Series(np.sort(data[x].unique()), name=x)
        if (
            aggregation_specifier.aggregation_func is AggregationType.COUNT
            or aggregation_specifier.aggregation_func
            is AggregationType.PROBABILITY
            or aggregation_specifier.aggregation_func
            is AggregationType.PERCENT
        ):
            _y_values: List = []
            for x_value in trace_data["x_values"]:
                y_value = (data[x] == x_value).sum()
                if (
                    aggregation_specifier.aggregation_func
                    is AggregationType.PROBABILITY
                ):
                    _y_values.append(y_value / data[x].notnull().sum())
                elif (
                    aggregation_specifier.aggregation_func
                    is AggregationType.PERCENT
                ):
                    _y_values.append(y_value / data[x].notnull().sum() * 100)
                else:
                    _y_values.append(y_value)
            trace_data["y_values"] = pd.Series(
                _y_values,
                name="_".join(
                    (x, aggregation_specifier.aggregation_func.value)
                ),
            )

        else:
            agg_func: Callable
            match aggregation_specifier.aggregation_func:
                case AggregationType.MEAN:
                    agg_func = np.mean
                case AggregationType.MEDIAN:
                    agg_func = np.median
                case AggregationType.SUM:
                    agg_func = np.sum

            trace_data["y_values"] = data.groupby(x)[y].apply(agg_func)

            if aggregation_specifier.error_bar is not None:
                trace_data["error_y"] = cls._compute_error_bar(
                    data.groupby(x)[y],
                    agg_func,
                    aggregation_specifier.error_bar,
                )

        return trace_data

    @classmethod
    def build_trace_data(
        cls, data: pd.DataFrame, aggregation_specifier: AggregationSpecifier
    ) -> "AggregationTraceData":
        trace_data = cls._build_aggregation_data_from_pointer(
            data, aggregation_specifier
        )

        return cls.parse_obj(trace_data)
