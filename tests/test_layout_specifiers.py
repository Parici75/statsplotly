import logging

import numpy as np
import pandas as pd
import pytest

from statsplotly.plot_specifiers.data import DataPointer, TraceData
from statsplotly.plot_specifiers.layout import (
    AxesSpecifier,
    LegendSpecifier,
    set_horizontal_colorbar,
)

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


logging.getLogger().setLevel(logging.DEBUG)


class TestAxesSpecifier:
    legend_specifier = LegendSpecifier(data_pointer=DataPointer(x="x", y="y", text="z"))

    def test_invalid_range(self):
        trace_data = TraceData.build_trace_data(
            data=EXAMPLE_DATAFRAME, pointer=DataPointer(x="x", y="y", text="z")
        )

        with pytest.raises(ValueError) as excinfo:
            AxesSpecifier(
                axis_format="square",
                traces=[trace_data],
                legend=self.legend_specifier,
                x_range=["a", "b"],
            )
            assert "Value error, Axis range must be numeric or `datetime`" in str(excinfo.value)

    def test_equal_range(self):
        trace_data = TraceData.build_trace_data(
            data=EXAMPLE_DATAFRAME.assign(x=np.arange(6, 9)),
            pointer=DataPointer(x="x", y="y", text="z"),
        )

        axes_specifier = AxesSpecifier(
            axis_format="equal", traces=[trace_data], legend=self.legend_specifier
        )
        assert axes_specifier.yaxis_range == [0.0, 8.8]

    def test_datetime_range(self, caplog):
        trace_data = TraceData.build_trace_data(
            data=EXAMPLE_DATETIME_DATAFRAME,
            pointer=DataPointer(x="x", y="y", text="z"),
        )
        axes_specifier = AxesSpecifier(
            axis_format="equal", traces=[trace_data], legend=self.legend_specifier
        )
        assert axes_specifier.yaxis_range == [
            np.datetime64("2020-01-01T00:00:00.000000000"),
            np.datetime64("2020-01-04T00:00:00.000000000"),
        ]
        assert "Can not pad a common range for values of type = 'datetime64[ns]'" in caplog.text

    def test_incompatible_axes(self, caplog):
        trace_data = TraceData.build_trace_data(
            data=EXAMPLE_DATAFRAME.assign(y=pd.date_range("2020-01-01", "2020-01-03", freq="D")),
            pointer=DataPointer(x="x", y="y", text="z"),
        )
        axes_specifier = AxesSpecifier(
            axis_format="equal", traces=[trace_data], legend=self.legend_specifier
        )
        assert axes_specifier.xaxis_range is None
        assert "Can not calculate a common range for values of type = 'object'" in caplog.text


def test_set_horizontal_colorbar():
    updated_colorbar = set_horizontal_colorbar(
        {
            "title": "dummy_title",
            "len": 1,
            "xanchor": "left",
            "yanchor": "middle",
            "tickmode": "auto",
        }
    )
    assert all(
        updated_colorbar[key] == value
        for key, value in {
            "orientation": "h",
            "y": -0.5,
            "xanchor": "center",
            "yanchor": "bottom",
        }.items()
    )
