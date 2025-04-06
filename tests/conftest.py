import numpy as np
import pandas as pd
import pytest

from statsplotly.plot_specifiers.data import DataHandler, DataPointer, TraceData
from statsplotly.plot_specifiers.layout import LegendSpecifier

_EXAMPLE_INPUT_DATA_DICT = {"x": ["a", "b", "c"], "y": range(3), "z": range(3)}

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
def example_input_data_dict():
    return _EXAMPLE_INPUT_DATA_DICT


@pytest.fixture(scope="module")
def example_input_dataframe():
    return pd.DataFrame(_EXAMPLE_INPUT_DATA_DICT)


@pytest.fixture(scope="module")
def example_input_datetime_dataframe():
    return _EXAMPLE_DATETIME_DATAFRAME


@pytest.fixture(scope="module")
def example_data_handler():
    return DataHandler.build_handler(
        data=_EXAMPLE_INPUT_DATA_DICT, data_pointer=DataPointer(x="x", y="y", text="z")
    )


@pytest.fixture(scope="module")
def example_trace_data(example_data_handler):
    return TraceData.build_trace_data(
        data=example_data_handler.data, pointer=example_data_handler.data_pointer
    )


@pytest.fixture(scope="module")
def example_3dtrace_data():
    handler = DataHandler.build_handler(
        data=_EXAMPLE_INPUT_DATA_DICT, data_pointer=DataPointer(x="x", y="y", z="z")
    )
    return TraceData.build_trace_data(
        data=handler.data,
        pointer=handler.data_pointer,
    )


@pytest.fixture(scope="module")
def example_datetime_trace_data():
    return TraceData.build_trace_data(
        data=_EXAMPLE_DATETIME_DATAFRAME, pointer=DataPointer(x="x", y="y", text="z")
    )


@pytest.fixture(scope="module")
def example_legend():
    return LegendSpecifier(data_pointer=DataPointer(x="x", y="y", text="z"))
