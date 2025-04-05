import logging

import numpy as np
import pandas as pd
import pytest

from statsplotly import constants
from statsplotly.plot_specifiers.data import (
    AggregationSpecifier,
    AggregationTraceData,
    DataDimension,
    DataHandler,
    DataPointer,
    DataProcessor,
    DataTypes,
    NormalizationType,
    TraceData,
)
from statsplotly.plot_specifiers.data.statistics import sem

logging.getLogger().setLevel(logging.DEBUG)


class TestDataPointer:
    def test_valid_pointer(self):
        assert DataPointer(x="x").y is None

    def test_missing_dimension_pointer(self):
        with pytest.raises(ValueError) as excinfo:
            DataPointer(z="z")
        assert "Both `x` and `y` dimensions can not be `None`" in str(excinfo.value)


class TestDataHandler:
    def test_array_input(self):
        data_handler = DataHandler.build_handler(
            data=np.array((2, 2)), data_pointer=DataPointer(x="0")
        )
        assert all(data_handler.get_data("x") == pd.Series(np.array((2, 2))))

    def test_no_slicer(self, example_input_data_dict):
        data_handler = DataHandler.build_handler(
            data=example_input_data_dict, data_pointer=DataPointer(x="x", y="y")
        )
        assert data_handler.n_slices == 1
        assert data_handler.slice_levels == []

    def test_slice_reorder(self, example_input_data_dict):
        data_handler = DataHandler.build_handler(
            data=example_input_data_dict,
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
            slice_order=[0, 2, 1],
        )
        assert data_handler.n_slices == 3
        assert (data_handler.get_data("y").to_numpy() == np.arange(3)).all()
        assert data_handler.slice_levels == [str(x) for x in [0, 2, 1]]
        assert [level for level, trace in list(data_handler.iter_slices())] == ["0", "2", "1"]

    def test_slice_partition(self, example_input_data_dict, caplog):
        data_handler = DataHandler.build_handler(
            data=example_input_data_dict,
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
            slice_order=[0, 1],
        )
        assert data_handler.n_slices == 2
        assert (data_handler.get_data("y").to_numpy() == np.arange(3)).all()
        assert data_handler.slice_levels == [str(x) for x in [0, 1]]
        assert [level for level, trace in list(data_handler.iter_slices())] == ["0", "1"]
        assert "[2] slices are not present in slices [0, 1] and will not be plotted" in caplog.text

    def test_invalid_slice_order(self, example_input_data_dict):
        with pytest.raises(
            ValueError,
            match="Invalid slice identifier: 'non_existing_slice_id' could not be found in 'z'",
        ):
            DataHandler.build_handler(
                data=example_input_data_dict,
                data_pointer=DataPointer(x="x", y="y", slicer="z"),
                slice_order=[0, 1, "non_existing_slice_id"],
            )

    def test_data_types(self, example_input_data_dict, example_input_dataframe):
        data_handler = DataHandler.build_handler(
            data=example_input_data_dict,
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
        )
        assert all(
            example_input_dataframe.dtypes.to_dict()[getattr(data_handler.data_pointer, key)] == val
            for (key, val) in data_handler.data_types.model_dump().items()
            if val is not None
        )

    def test_categorical_dtype_cast(self, example_input_dataframe, caplog):
        data_handler = DataHandler.build_handler(
            data=example_input_dataframe.assign(x=example_input_dataframe["x"].astype("category")),
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
        )
        assert data_handler.data_types.x is np.dtype("object")
        assert "Casting categorical 'x' data to string" in caplog.text

    def test_datetime_dtype(self, example_input_datetime_dataframe):
        data_handler = DataHandler.build_handler(
            data=example_input_datetime_dataframe,
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
        )
        assert data_handler.data_types.x == np.dtype("datetime64[ns]")

    def test_slicer_groupby_mean_aggregation(self, example_input_data_dict):
        agg_df = DataHandler.build_handler(
            data=example_input_data_dict,
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
        ).get_mean(dimension="y")
        assert agg_df.columns.tolist() == ["mean", "std"]
        assert agg_df.index.tolist() == [0, 1, 2]

    def test_no_groupby_median_aggregation(self, example_input_data_dict):
        agg_df = DataHandler.build_handler(
            data=example_input_data_dict,
            data_pointer=DataPointer(x="x", y="y"),
        ).get_median(dimension="y")
        assert agg_df.columns.tolist() == ["median", "iqr"]
        assert len(agg_df.index.tolist()) == 1

    def test_invalid_pointer(self, example_input_data_dict):
        with pytest.raises(ValueError, match=rf"u is not present in Index\("):
            DataHandler.build_handler(
                data=example_input_data_dict,
                data_pointer=DataPointer(x="u"),
                slice_order=None,
            )

    def test_invalid_dataframe(self):
        with pytest.raises(
            ValueError,
            match="Multi-indexed columns are not supported, flatten the header before calling statsplotly",
        ):
            DataHandler.build_handler(
                data=pd.DataFrame(columns=pd.MultiIndex.from_arrays((np.arange(3), np.arange(3)))),
                data_pointer=DataPointer(x="x", y="y"),
            )

    def test_empty_data(self):
        data = pd.DataFrame(columns=["x", "y", "z"])
        data_pointer = DataPointer(x="x", y="y", slicer="z")
        data_handler = DataHandler.build_handler(data=data, data_pointer=data_pointer)
        assert data_handler.n_slices == 0
        assert data_handler.data.empty


class TestDataProcessor:
    def test_valid_processor(self):
        data_processor = DataProcessor(
            jitter_settings={DataDimension.X: 0.2},
            normalizer={DataDimension.X: "zscore"},
        )
        assert len(data_processor.normalizer) == 1
        assert data_processor.normalizer[DataDimension.X] is NormalizationType.ZSCORE

    def test_invalid_normalizer(self):
        with pytest.raises(
            ValueError,
            match="1 validation error for DataProcessor\nnormalizer.x\n  Input should be 'center', 'minmax' or 'zscore'",
        ):
            DataProcessor(
                jitter_settings={DataDimension.X: 0.2},
                normalizer={DataDimension.X: "awesome_scaling"},
            )

    def test_unnormalizable_data(self, example_input_dataframe, caplog):
        data_processor = DataProcessor(
            normalizer={DataDimension.X: "zscore"},
        )
        processed_data = data_processor.process_trace_data({"x_values": example_input_dataframe.x})
        assert (processed_data["x_values"] == example_input_dataframe.x).all()
        assert (
            f"Dimension {DataDimension.X.value} of type {example_input_dataframe.x.dtype} can not be"
            f" normalized with {NormalizationType.ZSCORE.value}" in caplog.text
        )

    def test_unjitterable_data(self, example_input_dataframe, caplog):
        data_processor = DataProcessor(
            jitter_settings={DataDimension.X: 0.2},
        )
        processed_data = data_processor.process_trace_data({"x_values": example_input_dataframe.x})
        assert (processed_data["x_values"] == example_input_dataframe.x).all()
        assert (
            f"Dimension {DataDimension.X.value} of type {example_input_dataframe.x.dtype} can not be"
            " jittered" in caplog.text
        )

    def test_normalize_data(self, example_input_dataframe):
        data_processor = DataProcessor(
            normalizer={DataDimension.Y: "zscore"},
        )
        processed_data = data_processor.process_trace_data({"y_values": example_input_dataframe.y})
        assert processed_data["y_values"].mean() == 0


class TestTraceData:
    def test_build_trace_data(self, example_data_handler):
        trace_data = TraceData.build_trace_data(
            data=example_data_handler.data, pointer=example_data_handler.data_pointer
        )
        assert (trace_data.x_values == example_data_handler.data.x).all()
        assert (trace_data.y_values == example_data_handler.data.y).all()
        assert trace_data.z_values is None
        assert (
            trace_data.text_data
            == example_data_handler.data.z.apply(
                lambda x: f"{example_data_handler.data.z.name}: {x}"
            )
        ).all()

    def test_invalid_error_data(self, example_data_handler):
        with pytest.raises(ValueError, match="x error data must be numeric"):
            TraceData.build_trace_data(
                data=example_data_handler.data,
                pointer=DataPointer(x="x", y="y", error_x="x"),
            )

        with pytest.raises(
            ValueError,
            match="error_z error data must be bidirectional to be plotted relative to the underlying data",
        ):
            TraceData.build_trace_data(
                data=example_data_handler.data.copy().assign(
                    error_z=example_data_handler.data.z.apply(lambda x: [x])
                ),
                pointer=DataPointer(y="y", z="z", error_z="error_z"),
            )


class TestAggregationSpecifier:
    def test_mean_agg_specifier(self):
        agg_specifier = AggregationSpecifier(
            aggregation_func="mean",
            aggregated_dimension="y",
            data_pointer=DataPointer(x="x", y="y"),
            data_types=DataTypes(x=np.dtype(float), y=np.dtype(float)),
        )
        assert agg_specifier.reference_dimension is DataDimension.X
        assert agg_specifier.reference_data == "x"
        assert agg_specifier.aggregated_dimension is DataDimension.Y
        assert agg_specifier.aggregated_data == "y"
        assert agg_specifier.aggregation_plot_dimension is DataDimension.Y

    def test_no_agg_dimension(self):
        with pytest.raises(ValueError) as excinfo:
            AggregationSpecifier(
                aggregation_func="mean",
                aggregated_dimension="y",
                data_pointer=DataPointer(x="x"),
                data_types=DataTypes(x=np.dtype(float), y=np.dtype(float)),
            )
            assert "aggregation dimension `y` not found in the data"

    def test_count_agg_specifier(self):
        agg_specifier = AggregationSpecifier(
            aggregation_func="count",
            aggregated_dimension="x",
            data_pointer=DataPointer(x="x"),
            data_types=DataTypes(x=np.dtype(float)),
        )
        assert agg_specifier.reference_data == "x"
        assert agg_specifier.aggregation_plot_dimension is DataDimension.Y

    def test_invalid_count_agg_specifier(self):
        with pytest.raises(ValueError) as excinfo:
            AggregationSpecifier(
                aggregation_func="count",
                data_pointer=DataPointer(x="x", y="y"),
                data_types=DataTypes(x=np.dtype(float), y=np.dtype(float)),
            )
            assert "count aggregation only applies to one dimension" in str(excinfo.value)


class TestAggregationTraceData:
    def test_build_trace_data_with_sem(self, example_data_handler):
        trace_data = AggregationTraceData.build_aggregation_trace_data(
            data=example_data_handler.data,
            aggregation_specifier=AggregationSpecifier(
                aggregation_func="mean",
                aggregated_dimension="y",
                error_bar="sem",
                data_pointer=DataPointer(x="x", y="y"),
                data_types=example_data_handler.data_types,
            ),
        )
        assert (trace_data.y_values == example_data_handler.data.groupby("x")["y"].mean()).all()
        assert (
            trace_data.error_y.tolist()
            == example_data_handler.data.groupby("x")["y"]
            .apply(
                lambda x: (
                    x.mean() - sem(x, 1 - constants.CI_ALPHA),
                    (x.mean() + sem(x, 1 - constants.CI_ALPHA)),
                )
            )
            .tolist()
        )

    def test_build_count_trace_data(self, example_data_handler):
        trace_data = AggregationTraceData.build_aggregation_trace_data(
            data=example_data_handler.data,
            aggregation_specifier=AggregationSpecifier(
                aggregation_func="count",
                aggregated_dimension="y",
                data_pointer=DataPointer(y="y"),
                data_types=example_data_handler.data_types,
            ),
        )
        assert (
            trace_data.x_values.tolist() == example_data_handler.data.groupby("y")["y"].count()
        ).all()
        assert (trace_data.y_values.tolist() == example_data_handler.data["y"].unique()).all()

    def test_build_percent_trace_data(self, example_data_handler):
        trace_data = AggregationTraceData.build_aggregation_trace_data(
            data=example_data_handler.data,
            aggregation_specifier=AggregationSpecifier(
                aggregation_func="percent",
                aggregated_dimension="y",
                data_pointer=DataPointer(y="y"),
                data_types=example_data_handler.data_types,
            ),
        )
        assert (
            trace_data.x_values.tolist()
            == example_data_handler.data.groupby("y")["y"].count()
            / example_data_handler.data["y"].notnull().sum()
            * 100
        ).all()
        assert (trace_data.y_values.tolist() == example_data_handler.data["y"].unique()).all()
