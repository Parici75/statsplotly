import logging

import numpy as np
import pandas as pd
import pytest

from statsplot.plot_specifiers.data import TraceData, DataPointer
from statsplot.plot_specifiers.layout import AxesSpecifier, LegendSpecifier

EXAMPLE_DATAFRAME = pd.DataFrame(
    zip(["a", "b", "c"], np.arange(3), np.arange(3)),
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
        assert axes_specifier.yaxis_range == [0, 8]
