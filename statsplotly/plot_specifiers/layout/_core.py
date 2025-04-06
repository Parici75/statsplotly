from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
from dateutil.parser import parse as parse_date
from pydantic import BaseModel, field_validator

from statsplotly import constants
from statsplotly.plot_specifiers.common import smart_legend, smart_title
from statsplotly.plot_specifiers.data import (
    AggregationType,
    DataDimension,
    DataPointer,
    ErrorBarType,
    HistogramNormType,
    TraceData,
)

logger = logging.getLogger(__name__)


class PlotAxis(str, Enum):
    XAXIS = "xaxis"
    YAXIS = "yaxis"
    COLORAXIS = "coloraxis"


class ColoraxisReference(str, Enum):
    MAIN_COLORAXIS = "coloraxis"


class HistogramBarMode(str, Enum):
    STACK = "stack"
    OVERLAY = "overlay"


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


class AxisType(str, Enum):
    TWO_DIMENSIONAL = "two_dimensional"
    THREE_DIMENSIONAL = "three_dimensional"


class LegendSpecifier(BaseModel):
    data_pointer: DataPointer
    x_transformation: AggregationType | HistogramNormType | None = None
    y_transformation: AggregationType | HistogramNormType | None = None
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    z_label: str | None = None
    error_bar: str | None = None
    shaded_error: str | None = None
    axis_type: AxisType | None = None

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
        return self.x_label or smart_legend(
            self._get_axis_title_from_dimension_pointer(DataDimension.X)
        )

    @property
    def yaxis_title(self) -> str:
        return self.y_label or smart_legend(
            self._get_axis_title_from_dimension_pointer(DataDimension.Y)
        )

    @property
    def zaxis_title(self) -> str | None:
        if self.data_pointer.z is None:
            return None
        return self.z_label or smart_legend(
            self._get_axis_title_from_dimension_pointer(DataDimension.Z)
        )

    @property
    def figure_title(self) -> str:  # noqa: C901, PLR0912
        if self.title is not None:
            return self.title

        x_bit_title = f"{self.data_pointer.x or ''} {self.x_transformation_legend or ''}"
        y_bit_title = f"{self.data_pointer.y or ''} {self.y_transformation_legend or ''}"
        title_bits = [bit.strip() for bit in [y_bit_title, x_bit_title] if bit.strip()]
        title = " vs ".join(title_bits)

        if self.data_pointer.z is not None:
            if self.axis_type is AxisType.THREE_DIMENSIONAL:
                title = f"{title} vs {self.data_pointer.z}"
            else:
                title = f"{title} {self.data_pointer.z}"

        if self.data_pointer.slicer is not None:
            title = f"{title} per {self.data_pointer.slicer}"
        if self.error_bar is not None:
            if self.error_bar in (ErrorBarType.SEM, ErrorBarType.BOOTSTRAP):
                title = f"{title} ({(1 - constants.CI_ALPHA) * 100}% CI {self.error_bar})"
            else:
                title = f"{title} ({self.error_bar})"
        if self.shaded_error is not None:
            title = f"{title} (± {self.shaded_error})"

        return smart_title(title)


class AxesSpecifier(BaseModel):
    axis_format: AxisFormat | None = None
    traces: list[TraceData]
    legend: LegendSpecifier
    x_range: list[float | str] | None = None
    y_range: list[float | str] | None = None
    z_range: list[float | str] | None = None

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

    @staticmethod
    def pad_axis_range(axis_range: list[Any], padding_factor: float) -> list[Any]:
        if axis_range[0] < 0:
            axis_range[0] *= 1 + padding_factor
        else:
            axis_range[0] *= 1 - padding_factor

        axis_range[1] *= 1 + padding_factor

        return axis_range

    def get_axes_range(self) -> list[Any] | None:
        values_span = np.concatenate(
            [
                data
                for trace in self.traces
                for data in [trace.x_values, trace.y_values, trace.z_values]
                if data is not None
            ]
        )
        axes_span = [
            axis_span
            for axis_span in [self.x_range, self.y_range, self.z_range]
            if axis_span is not None
        ]
        try:
            if len(axes_span) > 0:
                min_value, max_value = (
                    np.max([np.min(values_span), np.min(axes_span)]),
                    np.min([np.max(values_span), np.max(axes_span)]),
                )

            else:
                min_value, max_value = np.min(values_span), np.max(values_span)

        except TypeError:
            logger.debug(
                f"Can not calculate a common range for values of type = '{values_span.dtype}'"
            )
            return None

        else:
            try:
                return self.pad_axis_range(
                    [min_value, max_value], padding_factor=constants.RANGE_PADDING_FACTOR
                )

            except TypeError:
                logger.debug(
                    f"Can not pad a common range for values of type = '{values_span.dtype}'"
                )
                return [min_value, max_value]

    @property
    def height(self) -> int | None:
        if self.axis_format in (
            AxisFormat.SQUARE,
            AxisFormat.ID_LINE,
        ):
            return constants.AXES_HEIGHT
        return None

    @property
    def width(self) -> int | None:
        if self.axis_format in (
            AxisFormat.SQUARE,
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
        if self.axis_format in (AxisFormat.FIXED_RATIO, AxisFormat.EQUAL, AxisFormat.ID_LINE):
            return 1
        return None

    @property
    def scaleanchor(self) -> str | None:
        if self.axis_format in (AxisFormat.FIXED_RATIO, AxisFormat.EQUAL, AxisFormat.ID_LINE):
            return "x"
        return None
