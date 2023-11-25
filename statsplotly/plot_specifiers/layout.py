import logging
from enum import Enum
from typing import Any

import numpy as np
from dateutil.parser import parse as parse_date
from pydantic import BaseModel, field_validator, model_validator

from statsplotly import constants
from statsplotly.exceptions import StatsPlotInvalidArgumentError
from statsplotly.plot_specifiers.data import (
    AggregationType,
    DataDimension,
    DataPointer,
    ErrorBarType,
    HistogramNormType,
    TraceData,
)
from statsplotly.utils.layout_utils import smart_legend, smart_title

logger = logging.getLogger(__name__)


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
    x_transformation: AggregationType | HistogramNormType | None = None
    y_transformation: AggregationType | HistogramNormType | None = None
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    z_label: str | None = None
    error_bar: str | None = None

    @model_validator(mode="before")
    def check_y_label(cls, values: dict[str, Any]) -> dict[str, Any]:
        data_pointer, y_transformation, y_label = (
            values.get("data_pointer"),
            values.get("y_transformation"),
            values.get("y_label"),
        )
        if data_pointer is None:
            raise ValueError("`data_pointer` can not be `None`")
        if data_pointer.y is None and y_transformation is None and y_label is None:
            raise ValueError(
                "No y_label provided for the legend, check"
                f" {LegendSpecifier.__name__} specification"
            )
        return values

    def _get_axis_title_from_dimension_pointer(self, dimension: DataDimension) -> str:
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
            self.x_label or self._get_axis_title_from_dimension_pointer(DataDimension.X)
        )

    @property
    def yaxis_title(self) -> str:
        return smart_legend(
            self.y_label or self._get_axis_title_from_dimension_pointer(DataDimension.Y)
        )

    @property
    def zaxis_title(self) -> str | None:
        if self.data_pointer.z is None:
            return None
        return smart_legend(
            self.z_label or self._get_axis_title_from_dimension_pointer(DataDimension.Z)
        )

    @property
    def figure_title(self) -> str:  # noqa: PLR0912
        if self.title is not None:
            return self.title

        if self.y_transformation is not None:
            if self.data_pointer.y is not None:
                title = (
                    f"{self.data_pointer.y} {self.y_transformation_legend} vs {self.data_pointer.x}"
                )
            else:
                title = f"{self.data_pointer.x} {self.y_transformation_legend}"
        elif self.x_transformation is not None:
            if self.data_pointer.x is not None:
                title = (
                    f"{self.data_pointer.x} {self.x_transformation_legend} vs {self.data_pointer.y}"
                )
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
    axis_format: AxisFormat | None = None
    traces: list[TraceData]
    legend: LegendSpecifier
    x_range: list[float | str] | None = None
    y_range: list[float | str] | None = None
    z_range: list[float | str] | None = None

    @field_validator("axis_format", mode="before")
    def validate_axis_format(cls, value: str | None) -> AxisFormat | None:
        try:
            return AxisFormat(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(value, AxisFormat) from exc  # type: ignore

    @field_validator("x_range", "y_range", "z_range")
    def validate_axis_range_format(
        cls, value: list[float | str] | None
    ) -> list[float | str] | None:
        if value is not None:
            try:
                [parse_date(limit) for limit in value if isinstance(limit, str)]
            except Exception as exc:
                raise ValueError("Axis range must be numeric or `datetime`") from exc
        return value

    def get_axes_range(self) -> list[Any] | None:
        values_span = np.concatenate([
            data
            for trace in self.traces
            for data in [trace.x_values, trace.y_values, trace.z_values]
            if data is not None
        ])
        axes_span = [
            axis_span
            for axis_span in [self.x_range, self.y_range, self.z_range]
            if axis_span is not None
        ]
        try:
            if len(axes_span) > 0:
                return [
                    np.max([np.min(values_span), np.min(axes_span)]),
                    np.min([np.max(values_span), np.max(axes_span)]),
                ]
            return [np.min(values_span), np.max(values_span)]
        except TypeError:
            logger.debug("Can not calculate a common range for axes")
            return None

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
    def xaxis_range(self) -> list[Any] | None:
        if self.axis_format in (AxisFormat.EQUAL, AxisFormat.ID_LINE):
            return self.get_axes_range()

        return self.x_range

    @property
    def yaxis_range(self) -> list[Any] | None:
        if self.axis_format in (AxisFormat.EQUAL, AxisFormat.ID_LINE):
            return self.get_axes_range()

        return self.y_range

    @property
    def zaxis_range(self) -> list[Any] | None:
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
        if self.axis_format in (AxisFormat.FIXED_RATIO, AxisFormat.EQUAL):
            return "x"
        return None
