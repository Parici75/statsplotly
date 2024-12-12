import datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from statsplotly.exceptions import StatsPlotSpecificationError
from statsplotly.plot_specifiers.data import (
    DataDimension,
    DataHandler,
    DataPointer,
    DataTypes,
    HistogramNormType,
    RegressionType,
)
from statsplotly.plot_specifiers.trace import (
    CategoricalPlotSpecifier,
    HistogramSpecifier,
    JointplotSpecifier,
    OrientedPlotSpecifier,
    PlotOrientation,
    ScatterSpecifier,
    TraceMode,
)


def test_scatter_specifier():
    scatter_specifier = ScatterSpecifier(
        mode="lines",
        regression_type="linear",
        data_types=DataTypes(x=np.dtype(float), y=np.dtype(float)),
    )
    assert scatter_specifier.mode is TraceMode.LINES
    assert scatter_specifier.regression_type is RegressionType.LINEAR

    scatter_specifier = ScatterSpecifier(
        mode=None,
        data_types=DataTypes(x=np.dtype(object), y=np.dtype(float)),
    )
    assert scatter_specifier.mode is TraceMode.MARKERS

    with pytest.raises(ValueError) as excinfo:
        ScatterSpecifier(mode="circles")
        assert (
            "Invalid value: 'circles'. Value must be one of"
            f" {[member.value for member in TraceMode]}" in str(excinfo.value)
        )

    with pytest.raises(ValueError) as excinfo:
        ScatterSpecifier(data_types=DataTypes(x=np.dtype(float)))
        assert "Both `x` and `y`dimensions must be supplied for ScatterSpecifier" in str(
            excinfo.value
        )


class TestOrientedPlotSpecifier:
    def test_default_vertical_plot(self):
        plot_specifier = OrientedPlotSpecifier(
            data_types=DataTypes(x=np.dtype(float), y=np.dtype(float))
        )
        assert plot_specifier.orientation is PlotOrientation.VERTICAL
        assert plot_specifier.anchor_dimension is DataDimension.X
        assert plot_specifier.anchored_dimension is DataDimension.Y

    def test_default_horizontal_plot(self):
        plot_specifier = OrientedPlotSpecifier(
            data_types=DataTypes(x=np.dtype(float), y=np.dtype(str))
        )
        assert plot_specifier.orientation is PlotOrientation.HORIZONTAL
        assert plot_specifier.anchor_dimension is DataDimension.Y
        assert plot_specifier.anchored_dimension is DataDimension.X

    def test_forced_horizontal_plot(self):
        plot_specifier = OrientedPlotSpecifier(
            data_types=DataTypes(x=np.dtype(float), y=np.dtype(float)),
            prefered_orientation="horizontal",
        )
        assert plot_specifier.orientation is PlotOrientation.HORIZONTAL


class TestCategoricalPlotSpecifier:
    def test_string_vertical_x_strip_map(self, example_raw_data):
        data_handler = DataHandler.build_handler(
            data=example_raw_data, data_pointer=DataPointer(x="x", y="y")
        )

        x_values_map = CategoricalPlotSpecifier(
            plot_type="stripplot",
            prefered_orientation="vertical",
            data_types=data_handler.data_types,
        ).get_category_strip_map(data_handler=data_handler)
        assert x_values_map == {
            DataDimension.X: dict(
                zip(
                    data_handler.get_data("x").sort_values(),
                    np.arange(len(data_handler.data)) + 1,
                    strict=False,
                )
            )
        }

    def test_datetime_horizontal_x_strip_map(self, example_raw_datetime_data):
        data_handler = DataHandler.build_handler(
            data=example_raw_datetime_data, data_pointer=DataPointer(x="x", y="y")
        )
        y_values_map = CategoricalPlotSpecifier(
            plot_type="stripplot",
            prefered_orientation="horizontal",
            data_types=data_handler.data_types,
        ).get_category_strip_map(data_handler=data_handler)
        assert y_values_map == {
            DataDimension.Y: dict(
                zip(
                    data_handler.get_data("y").sort_values().astype(str),
                    np.arange(len(data_handler.data)) + 1,
                    strict=False,
                )
            )
        }

    def test_numeric_data_no_x_strip_map(self, example_raw_data):
        data_handler = DataHandler.build_handler(
            data=example_raw_data, data_pointer=DataPointer(x="x", y="y")
        )

        x_values_map = CategoricalPlotSpecifier(
            plot_type="stripplot",
            prefered_orientation="horizontal",
            data_types=data_handler.data_types,
        ).get_category_strip_map(data_handler=data_handler)
        assert x_values_map is None

    def test_color_data_not_allowed(self, example_raw_data):
        data_handler = DataHandler.build_handler(
            data=example_raw_data, data_pointer=DataPointer(x="x", y="y", color="z")
        )
        with pytest.raises(ValueError) as excinfo:
            CategoricalPlotSpecifier(
                plot_type="boxplot",
                forced_orientation="vertical",
                data_types=data_handler.data_types,
            )
            assert (
                "Only slice-level color data can be specified with `boxplot`, got marker-level argument `color=z`"
                in str(excinfo.value)
            )


class TestHistogramSpecifier:
    bins = 10

    def test_invalid_histogram_parameters(self):
        with pytest.raises(ValueError) as excinfo:
            HistogramSpecifier(
                hist=False,
                cumulative=True,
                bins=None,
                dimension=DataDimension.X,
                histnorm=None,
                data_type=np.dtype("int"),
            )
        assert "Cumulative histogram requires histogram bins plotting" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            HistogramSpecifier(
                hist=True,
                bins=None,
                kde=True,
                cumulative=True,
                dimension=DataDimension.X,
                histnorm=None,
                data_type=np.dtype("int"),
            )
        assert "KDE is incompatible with cumulative histogram plotting" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            HistogramSpecifier(
                bins=None,
                kde=True,
                dimension=DataDimension.X,
                histnorm="",
                data_type=np.dtype("int"),
            )
        assert (
            "Histogram norm must be set to probability density with KDE plotting, got `COUNT`"
            in str(excinfo.value)
        )

    def test_automatic_histnorm(self):
        histogram_specifier = HistogramSpecifier(
            bins=None,
            kde=True,
            dimension=DataDimension.X,
            histnorm=None,
            data_type=np.dtype("int"),
        )
        assert histogram_specifier.histnorm is HistogramNormType.PROBABILITY_DENSITY

    def test_histogram_bin_edges(self):
        histogram_specifier = HistogramSpecifier(
            hist=True,
            dimension=DataDimension.X,
            histnorm=None,
            bins=self.bins,
            data_type=np.dtype("int"),
        )
        assert not histogram_specifier.density
        assert histogram_specifier.bins == self.bins
        bin_edges, bin_size = histogram_specifier.get_histogram_bin_edges(pd.Series(np.arange(100)))
        assert len(bin_edges) == self.bins + 1
        assert bin_size == 9.9

    @pytest.mark.parametrize(
        "cumulative_option, hist_expected", [(False, 10), (True, np.cumsum(np.repeat(10, 10)))]
    )
    def test_compute_histogram(self, cumulative_option, hist_expected):
        histogram_specifier = HistogramSpecifier(
            hist=True,
            cumulative=cumulative_option,
            dimension=DataDimension.X,
            histnorm=None,
            bins=self.bins,
            data_type=np.dtype("int"),
        )
        hist, bin_edges, bin_size = histogram_specifier.compute_histogram(pd.Series(np.arange(100)))
        assert all(hist == hist_expected)

    def test_compute_ecdf(self):
        histogram_specifier = HistogramSpecifier(
            hist=False,
            dimension=DataDimension.X,
            histnorm=None,
            bins=None,
            data_type=np.dtype("int"),
        )
        sample_data = [6.23, 5.58, 7.06, 6.42, 5.20]
        ranks, unique_values = histogram_specifier.compute_ecdf(pd.Series(sample_data))
        assert all(ranks == pd.Series(np.arange(len(sample_data))) + 1)
        assert all(unique_values == np.sort(sample_data))


class TestJointplotSpecifier:

    def test_jointplot_specifier(self, caplog):
        jointplot_specifier = JointplotSpecifier(
            plot_type="kde",
            marginal_plot="all",
            scatter_specifier=ScatterSpecifier(
                mode="lines",
                data_types=DataTypes(x=np.dtype(float), y=np.dtype(float)),
            ),
            histogram_specifier={
                DataDimension.X: HistogramSpecifier(
                    hist=True,
                    dimension=DataDimension.X,
                    histnorm="",
                    bins="scott",
                    data_type=np.dtype("int"),
                ),
                DataDimension.Y: HistogramSpecifier(
                    hist=True,
                    dimension=DataDimension.Y,
                    histnorm="",
                    bins="scott",
                    data_type=np.dtype("int"),
                ),
            },
        )
        assert jointplot_specifier.plot_kde
        assert not jointplot_specifier.plot_scatter

        with pytest.raises(ValidationError) as excinfo:
            JointplotSpecifier(
                plot_type="kde",
                marginal_plot="all",
                scatter_specifier=ScatterSpecifier(
                    mode="lines",
                    regression_type="linear",
                    data_types=DataTypes(x=np.dtype(float), y=np.dtype(float)),
                ),
            )
            assert ("linear regression can not be displayed on a JointplotType.KDE plot") in str(
                excinfo
            )

    def test_histogram2d(self):
        bins = 10
        jointplot_specifier = JointplotSpecifier(
            plot_type="kde",
            marginal_plot="all",
            scatter_specifier=ScatterSpecifier(
                mode="lines", data_types=DataTypes(x=np.dtype(float), y=np.dtype(float))
            ),
            histogram_specifier={
                DataDimension.X: HistogramSpecifier(
                    hist=True,
                    dimension=DataDimension.X,
                    histnorm="",
                    bins=10,
                    data_type=np.dtype("int"),
                ),
                DataDimension.Y: HistogramSpecifier(
                    hist=True,
                    dimension=DataDimension.Y,
                    histnorm="",
                    bins=10,
                    data_type=np.dtype("int"),
                ),
            },
        )
        hist, bin_edges, bin_size = jointplot_specifier.histogram2d(
            pd.DataFrame((np.arange(100), np.arange(100)))
        )
        assert len(hist) == bins**2
        assert len(bin_edges) == 2
        assert len(bin_size) == 2
