import numpy as np
import pandas as pd
import pytest

from statsplotly.plot_specifiers.data import DataPointer, TraceData
from statsplotly.plot_specifiers.layout import LegendSpecifier

EXAMPLE_DATAFRAME = pd.DataFrame(
    zip(["a", "b", "c"], np.arange(3), np.arange(3), strict=False),
    columns=["x", "y", "z"],
)

EXAMPLE_DATETIME_DATAFRAME = pd.DataFrame(
    zip(
        pd.date_range("2020-01-01", "2020-01-03", freq="D"),
        pd.date_range("2020-01-02", "2020-01-04", freq="D"),
        np.arange(3),
        strict=False,
    ),
    columns=["x", "y", "z"],
)


@pytest.fixture(scope="module")
def example_trace_data():
    return TraceData.build_trace_data(
        data=EXAMPLE_DATAFRAME, pointer=DataPointer(x="x", y="y", text="z")
    )


@pytest.fixture(scope="module")
def example_3dtrace_data():
    return TraceData.build_trace_data(
        data=EXAMPLE_DATAFRAME, pointer=DataPointer(x="x", y="y", z="z")
    )


@pytest.fixture(scope="module")
def example_datetime_trace_data():
    return TraceData.build_trace_data(
        data=EXAMPLE_DATETIME_DATAFRAME, pointer=DataPointer(x="x", y="y", text="z")
    )


@pytest.fixture(scope="module")
def example_legend():
    return LegendSpecifier(data_pointer=DataPointer(x="x", y="y", text="z"))
