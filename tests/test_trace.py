import numpy as np
import pandas as pd
import pytest

from statsplotly.plot_specifiers.trace import (
    DataDimension,
    HistogramSpecifier,
    JointplotSpecifier,
    RegressionType,
    ScatterSpecifier,
    TraceMode,
)


def test_scatter_specifier():
    scatter_specifier = ScatterSpecifier(mode="lines", regression_type="linear")
    assert scatter_specifier.mode is TraceMode.LINES
    assert scatter_specifier.regression_type is RegressionType.LINEAR

    with pytest.raises(ValueError) as excinfo:
        ScatterSpecifier(mode="circles")
        assert (
            "Invalid value: 'circles'. Value must be one of"
            f" {[member.value for member in TraceMode]}" in str(excinfo.value)
        )


class TestHistogramSpecifier:
    def test_invalid_histogram_parameters(self):
        with pytest.raises(ValueError) as excinfo:
            HistogramSpecifier(
                cumulative=True,
                dimension=DataDimension.X,
                histnorm="",
                data_type=np.dtype("int"),
            )
        assert "Cumulative histogram requires histogram bins plotting" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            HistogramSpecifier(
                hist=True,
                kde=True,
                cumulative=True,
                dimension=DataDimension.X,
                histnorm="",
                data_type=np.dtype("int"),
            )
        assert "KDE is incompatible with cumulative histogram plotting" in str(excinfo.value)

    def test_histogram_bin_edges(self):
        bins = 10
        histogram_specifier = HistogramSpecifier(
            hist=True,
            dimension=DataDimension.X,
            histnorm="",
            bins=bins,
            data_type=np.dtype("int"),
        )
        assert not histogram_specifier.density
        assert histogram_specifier.bins == bins
        bin_edges, bin_size = histogram_specifier.histogram_bin_edges(pd.Series(np.arange(100)))
        assert len(bin_edges) == bins + 1
        assert bin_size == 9.9

    def test_histogram(self):
        bins = 10
        histogram_specifier = HistogramSpecifier(
            hist=True,
            dimension=DataDimension.X,
            histnorm="",
            bins=bins,
            data_type=np.dtype("int"),
        )
        hist, bin_edges, bin_size = histogram_specifier.histogram(pd.Series(np.arange(100)))
        assert (hist == 10).all()


class TestJointplotSpecifier:
    scatter_specifier = ScatterSpecifier(mode="lines", regression_type="linear")

    def test_jointplot_specifier(self):
        jointplot_specifier = JointplotSpecifier(
            plot_type="kde",
            marginal_plot="all",
            scatter_specifier=ScatterSpecifier(),
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

    def test_histogram2d(self):
        bins = 10
        jointplot_specifier = JointplotSpecifier(
            plot_type="kde",
            marginal_plot="all",
            scatter_specifier=ScatterSpecifier(),
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
