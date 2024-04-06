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
    NormalizationType,
    TraceData,
)
from statsplotly.plot_specifiers.data.statistics import sem

EXAMPLE_DATAFRAME = pd.DataFrame(
    zip(["a", "b", "c"], np.arange(3), np.arange(3), strict=False),
    columns=["x", "y", "z"],
)

logging.getLogger().setLevel(logging.DEBUG)


class TestDataPointer:
    def test_valid_pointer(self):
        assert DataPointer(x="x").y is None

    def test_missing_dimension_pointer(self):
        with pytest.raises(ValueError) as excinfo:
            DataPointer(z="z")
        assert "Both x and y dimensions can not be None" in str(excinfo.value)


class TestDataHandler:
    def test_slice_reorder(self):
        data_handler = DataHandler.build_handler(
            data=EXAMPLE_DATAFRAME,
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
            slice_order=[0, 2, 1],
        )
        assert data_handler.n_slices == 3
        assert (data_handler.get_data("y").values == np.arange(3)).all()
        assert data_handler.slice_levels == [str(x) for x in [0, 2, 1]]
        assert [level for level, trace in list(data_handler.iter_slices())] == ["0", "2", "1"]

    def test_slice_partition(self, caplog):
        data_handler = DataHandler.build_handler(
            data=EXAMPLE_DATAFRAME,
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
            slice_order=[0, 1],
        )
        assert data_handler.n_slices == 2
        assert (data_handler.get_data("y").values == np.arange(3)).all()
        assert data_handler.slice_levels == [str(x) for x in [0, 1]]
        assert [level for level, trace in list(data_handler.iter_slices())] == ["0", "1"]
        assert "[2] slices are not present in slices [0, 1] and will not be plotted" in caplog.text

    def test_data_types(self):
        data_handler = DataHandler.build_handler(
            data=EXAMPLE_DATAFRAME,
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
        )
        assert all(
            EXAMPLE_DATAFRAME.dtypes.to_dict()[getattr(data_handler.data_pointer, key)] == val
            for (key, val) in data_handler.data_types.model_dump().items()
            if val is not None
        )

    def test_categorical_dtype_cast(self, caplog):
        data_handler = DataHandler.build_handler(
            data=EXAMPLE_DATAFRAME.assign(x=EXAMPLE_DATAFRAME["x"].astype("category")),
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
        )
        assert data_handler.data_types.x is np.dtype("object")
        assert "Casting categorical x data to string" in caplog.text

    def test_slicer_groupby_mean_aggregation(self):
        agg_df = DataHandler.build_handler(
            data=EXAMPLE_DATAFRAME,
            data_pointer=DataPointer(x="x", y="y", slicer="z"),
        ).get_mean(dimension="y")
        assert agg_df.columns.tolist() == ["mean", "std"]
        assert agg_df.index.tolist() == [0, 1, 2]

    def test_no_groupby_median_aggregation(self):
        agg_df = DataHandler.build_handler(
            data=EXAMPLE_DATAFRAME,
            data_pointer=DataPointer(x="x", y="y"),
        ).get_median(dimension="y")
        assert agg_df.columns.tolist() == ["median", "iqr"]
        assert len(agg_df.index.tolist()) == 1

    def test_invalid_pointer(self):
        with pytest.raises(ValueError) as excinfo:
            DataHandler.build_handler(
                data=EXAMPLE_DATAFRAME,
                data_pointer=DataPointer(x="u"),
                slice_order=None,
            )
            assert f"u is not present in {EXAMPLE_DATAFRAME.columns}" in str(excinfo.value)

    def test_invalid_dataframe(self, caplog):
        with pytest.raises(ValueError):
            DataHandler.build_handler(
                data=pd.DataFrame(columns=pd.MultiIndex.from_arrays((np.arange(3), np.arange(3)))),
                data_pointer=DataPointer(),
                slice_order=None,
            )
            assert (
                "Multi-indexed columns are not supported, flatten the header before calling"
                " statsplotly" in caplog.text
            )


class TestDataProcessor:
    def test_valid_processor(self):
        data_processor = DataProcessor(
            x_values_mapping=dict(
                zip(EXAMPLE_DATAFRAME.x, np.arange(len(EXAMPLE_DATAFRAME.x)), strict=False)
            ),
            jitter_settings={DataDimension.X: 0.2},
            normalizer={DataDimension.X: "zscore"},
        )
        assert len(data_processor.normalizer) == 1
        assert data_processor.normalizer[DataDimension.X] is NormalizationType.ZSCORE

    def test_invalid_normalizer(self):
        with pytest.raises(ValueError) as excinfo:
            DataProcessor(
                x_values_mapping=dict(
                    zip(
                        EXAMPLE_DATAFRAME.x,
                        np.arange(len(EXAMPLE_DATAFRAME.x)),
                        strict=False,
                    )
                ),
                jitter_settings={DataDimension.X: 0.2},
                normalizer={DataDimension.X: "awesome_scaling"},
            )
        assert (
            "1 validation error for DataProcessor\nnormalizer.x\n  Input should be 'center', 'minmax' or 'zscore'"
            in str(excinfo.value)
        )

    def test_unnormalizable_data(self, caplog):
        data_processor = DataProcessor(
            normalizer={DataDimension.X: "zscore"},
        )
        processed_data = data_processor.process_trace_data({"x_values": EXAMPLE_DATAFRAME.x})
        assert (processed_data["x_values"] == EXAMPLE_DATAFRAME.x).all()
        assert (
            f"Dimension {DataDimension.X.value} of type {EXAMPLE_DATAFRAME.x.dtype} can not be"
            f" normalized with {NormalizationType.ZSCORE.value}" in caplog.text
        )

    def test_unjitterable_data(self, caplog):
        data_processor = DataProcessor(
            jitter_settings={DataDimension.X: 0.2},
        )
        processed_data = data_processor.process_trace_data({"x_values": EXAMPLE_DATAFRAME.x})
        assert (processed_data["x_values"] == EXAMPLE_DATAFRAME.x).all()
        assert (
            f"Dimension {DataDimension.X.value} of type {EXAMPLE_DATAFRAME.x.dtype} can not be"
            " jittered" in caplog.text
        )

    def test_normalize_data(self):
        data_processor = DataProcessor(
            normalizer={DataDimension.Y: "zscore"},
        )
        processed_data = data_processor.process_trace_data({"y_values": EXAMPLE_DATAFRAME.y})
        assert processed_data["y_values"].mean() == 0


class TestTraceData:
    def test_build_trace_data(self):
        trace_data = TraceData.build_trace_data(
            data=EXAMPLE_DATAFRAME, pointer=DataPointer(x="x", y="y", text="z")
        )
        assert (trace_data.x_values == EXAMPLE_DATAFRAME.x).all()
        assert (trace_data.y_values == EXAMPLE_DATAFRAME.y).all()
        assert trace_data.z_values is None
        assert (
            trace_data.text_data
            == EXAMPLE_DATAFRAME.z.apply(lambda x: f"{EXAMPLE_DATAFRAME.z.name}: {x}")
        ).all()

    def test_invalid_error_data(self):
        with pytest.raises(ValueError) as excinfo:
            TraceData.build_trace_data(
                data=EXAMPLE_DATAFRAME,
                pointer=DataPointer(x="x", y="y", error_x="x"),
            )
        assert "x error data must be numeric" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            TraceData.build_trace_data(
                data=EXAMPLE_DATAFRAME.copy().assign(
                    error_z=EXAMPLE_DATAFRAME.z.apply(lambda x: [x])
                ),
                pointer=DataPointer(y="y", z="z", error_z="error_z"),
            )
        assert (
            "error_z error data must be bidirectional to be plotted relative to the underlying data"
            in str(excinfo.value)
        )


class TestAggregationTraceData:
    def test_build_trace_data_with_sem(self):
        trace_data = AggregationTraceData.build_aggregation_trace_data(
            data=EXAMPLE_DATAFRAME,
            aggregation_specifier=AggregationSpecifier(
                aggregation_func="mean",
                error_bar="sem",
                data_pointer=DataPointer(x="x", y="y"),
                data_types=DataHandler(
                    data=EXAMPLE_DATAFRAME,
                    data_pointer=DataPointer(x="x", y="y"),
                ).data_types,
            ),
        )
        assert (trace_data.y_values == EXAMPLE_DATAFRAME.groupby("x")["y"].mean()).all()
        assert (
            trace_data.error_y.tolist()
            == EXAMPLE_DATAFRAME.groupby("x")["y"]
            .apply(
                lambda x: (
                    x.mean() - sem(x, 1 - constants.CI_ALPHA),
                    (x.mean() + sem(x, 1 - constants.CI_ALPHA)),
                )
            )
            .tolist()
        )

    def test_invalid_aggregation_dtypes(self):
        data_pointer = DataPointer(x="y", y="x")
        with pytest.raises(ValueError) as excinfo:
            AggregationSpecifier(
                aggregation_func="mean",
                error_bar="sem",
                data_pointer=data_pointer,
                data_types=DataHandler(
                    data=EXAMPLE_DATAFRAME,
                    data_pointer=data_pointer,
                ).data_types,
            )
        assert "mean aggregation requires numeric type y data, got: `object`" in str(excinfo.value)
