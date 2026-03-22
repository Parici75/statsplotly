from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest

from statsplotly.plot_specifiers.data import DataHandler, DataPointer, TraceData
from statsplotly.plot_specifiers.layout import LegendSpecifier

_EXAMPLE_INPUT_DATA_DICT = {"x": ["a", "b", "c"], "y": range(3), "z": range(3), "size": [8, 8, 8]}

_EXAMPLE_DATETIME_DATAFRAME = pd.DataFrame(
    zip(
        pd.date_range("2020-01-01", "2020-01-03", freq="D"),
        pd.date_range("2020-01-02", "2020-01-04", freq="D"),
        np.arange(3),
        strict=False,
    ),
    columns=["x", "y", "z"],
)


@pytest.fixture(scope="module")
def example_input_data_dict() -> dict[str, Any]:
    return _EXAMPLE_INPUT_DATA_DICT


@pytest.fixture(scope="module")
def dataframe_factory() -> Callable[[Literal["numpy", "pyarrow"]], pd.DataFrame]:
    def inner(backend: Literal["numpy", "pyarrow"] = "pyarrow") -> pd.DataFrame:
        return pd.DataFrame(_EXAMPLE_INPUT_DATA_DICT).convert_dtypes(
            dtype_backend=backend if backend == "pyarrow" else "numpy_nullable"
        )

    return inner


@pytest.fixture(scope="module")
def example_input_dataframe(dataframe_factory) -> pd.DataFrame:
    return dataframe_factory(backend="pyarrow")


@pytest.fixture(scope="module")
def example_input_datetime_dataframe() -> pd.DataFrame:
    return _EXAMPLE_DATETIME_DATAFRAME


@pytest.fixture(scope="module")
def example_data_handler() -> DataHandler:
    return DataHandler.build_handler(
        data=_EXAMPLE_INPUT_DATA_DICT, data_pointer=DataPointer(x="x", y="y", text="z", size="size")
    )


@pytest.fixture(scope="module")
def example_trace_data(example_data_handler: DataHandler) -> TraceData:
    return TraceData.build_from_data(
        data=example_data_handler.data, pointer=example_data_handler.data_pointer
    )


@pytest.fixture(scope="module")
def example_3dtrace_data() -> TraceData:
    handler = DataHandler.build_handler(
        data=_EXAMPLE_INPUT_DATA_DICT, data_pointer=DataPointer(x="x", y="y", z="z")
    )
    return TraceData.build_from_data(
        data=handler.data,
        pointer=handler.data_pointer,
    )


@pytest.fixture(scope="module")
def example_datetime_trace_data() -> TraceData:
    return TraceData.build_from_data(
        data=_EXAMPLE_DATETIME_DATAFRAME, pointer=DataPointer(x="x", y="y", text="z")
    )


@pytest.fixture(scope="module")
def example_legend() -> LegendSpecifier:
    return LegendSpecifier(data_pointer=DataPointer(x="x", y="y", text="z"))
