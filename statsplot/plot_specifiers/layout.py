from enum import Enum
from typing import Dict, Any, List, Sequence

import numpy as np
from pydantic import BaseModel, root_validator, validator

from statsplot import constants
from statsplot.exceptions import StatsPlotInvalidArgumentError
from statsplot.plot_specifiers.data import (
    DataPointer,
    AggregationType,
    DataDimension,
    TraceData,
    HistogramNormType,
    ErrorBarType,
)
from statsplot.utils.layout_utils import smart_legend, smart_title


class BarMode(str, Enum):
    STACK = "stack"
    GROUP = "group"
    OVERLAY = "overlay"
    RELATIVE = "relative"


class AxisFormat(str, Enum):
    SQUARE = "square"
    FIXED_RATIO = "fixed_ratio"
    EQUAL = "equal"
    ID_LINE = "id_line"


class ColoraxisReference(str, Enum):
    MAIN_COLORAXIS = "coloraxis"


class LegendSpecifier(BaseModel):
    data_pointer: DataPointer
    x_transformation: AggregationType | HistogramNormType | None
    y_transformation: AggregationType | HistogramNormType | None
    title: str | None
    x_label: str | None
    y_label: str | None
    z_label: str | None
    error_bar: str | None

    @root_validator(pre=True)
    def check_y_label(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        data_pointer, y_transformation, y_label = (
            values.get("data_pointer"),
            values.get("y_transformation"),
            values.get("y_label"),
        )
        assert data_pointer is not None
        if (
            data_pointer.y is None
            and y_transformation is None
            and y_label is None
        ):
            raise ValueError(
                f"No y_label provided for the legend, check {LegendSpecifier.__name__} specification"
            )
        return values

    def _get_axis_title_from_dimension_pointer(
        self, dimension: DataDimension
    ) -> str:
        pointer_label = getattr(self.data_pointer, dimension) or ""
        if dimension is DataDimension.X:
            return f"{pointer_label} {self.x_transformation_legend or ''}"
        if dimension is DataDimension.Y:
            return f"{pointer_label} {self.y_transformation_legend or ''}"
        return pointer_label

    @property
    def y_transformation_legend(self) -> str | None:
        if self.y_transformation is None:
            return None
        return (
            self.y_transformation.value
            if self.y_transformation is not HistogramNormType.COUNT
            else "count"
        )

    @property
    def x_transformation_legend(self) -> str | None:
        if self.x_transformation is None:
            return None
        return (
            self.x_transformation.value
            if self.x_transformation is not HistogramNormType.COUNT
            else "count"
        )

    @property
    def xaxis_title(self) -> str:
        return smart_legend(
            self.x_label
            or self._get_axis_title_from_dimension_pointer(DataDimension.X)
        )

    @property
    def yaxis_title(self) -> str:
        return smart_legend(
            self.y_label
            or self._get_axis_title_from_dimension_pointer(DataDimension.Y)
        )

    @property
    def zaxis_title(self) -> str | None:
        if self.data_pointer.z is None:
            return None
        return smart_legend(
            self.z_label
            or self._get_axis_title_from_dimension_pointer(DataDimension.Z)
        )

    @property
    def figure_title(self) -> str:
        if self.title is not None:
            return self.title

        if self.y_transformation is not None:
            if self.data_pointer.y is not None:
                title = f"{self.data_pointer.y} {self.y_transformation_legend} vs {self.data_pointer.x}"
            else:
                title = f"{self.data_pointer.x} {self.y_transformation_legend}"
        elif self.x_transformation is not None:
            if self.data_pointer.x is not None:
                title = f"{self.data_pointer.x} {self.x_transformation_legend} vs {self.data_pointer.y}"
            else:
                title = f"{self.data_pointer.y} {self.x_transformation_legend}"
        else:
            title = f"{self.data_pointer.y} vs {self.data_pointer.x}"

        if self.data_pointer.z is not None:
            title = f"{title} vs {self.data_pointer.z}"
        if self.data_pointer.slicer is not None:
            title = f"{title} per {self.data_pointer.slicer}"
        if self.error_bar is not None:
            if self.error_bar in (ErrorBarType.SEM, ErrorBarType.BOOTSTRAP):
                title = f"{title} ({(1 - constants.CI_ALPHA) * 100}% CI {self.error_bar})"
            else:
                title = f"{title} ({self.error_bar})"

        return smart_title(title)


class AxesSpecifier(BaseModel):
    axis_format: AxisFormat | None
    traces: List[TraceData]
    legend: LegendSpecifier
    x_range: Sequence[float] | None
    y_range: Sequence[float] | None
    z_range: Sequence[float] | None

    @validator("axis_format", pre=True)
    def validate_axis_format(cls, value: str | None) -> AxisFormat | None:
        try:
            return AxisFormat(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, AxisFormat  # type: ignore
            ) from exc

    def get_axes_range(self) -> List[float]:
        values_span = np.concatenate(
            [
                data
                for trace in self.traces
                for data in [trace.x_values, trace.y_values, trace.z_values]
                if data is not None
            ]
        )
        try:
            return [min(values_span), max(values_span)]
        except TypeError:
            return values_span[[0, -1]].tolist()

    @property
    def height(self) -> int | None:
        if self.axis_format in (
            AxisFormat.SQUARE,
            AxisFormat.EQUAL,
            AxisFormat.ID_LINE,
        ):
            return constants.AXES_HEIGHT
        return None

    @property
    def width(self) -> int | None:
        if self.axis_format in (
            AxisFormat.SQUARE,
            AxisFormat.EQUAL,
            AxisFormat.ID_LINE,
        ):
            return constants.AXES_WIDTH
        return None

    @property
    def xaxis_range(self) -> Sequence[float] | None:
        if self.axis_format in (AxisFormat.EQUAL, AxisFormat.ID_LINE):
            return self.get_axes_range()
        return self.x_range

    @property
    def yaxis_range(self) -> Sequence[float] | None:
        if self.axis_format in (AxisFormat.EQUAL, AxisFormat.ID_LINE):
            return self.get_axes_range()
        return self.y_range

    @property
    def zaxis_range(self) -> Sequence[float] | None:
        if self.axis_format is AxisFormat.EQUAL:
            return self.get_axes_range()
        return self.z_range

    @property
    def scaleratio(self) -> float | None:
        if self.axis_format in (AxisFormat.FIXED_RATIO, AxisFormat.EQUAL):
            return 1
        return None

    @property
    def scaleanchor(self) -> str | None:
        if (
            self.axis_format is AxisFormat.FIXED_RATIO
            or self.axis_format is AxisFormat.EQUAL
        ):
            return "x"
        return None
