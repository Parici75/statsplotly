from __future__ import annotations

import logging
from typing import Any

from pydantic import field_validator
from pydantic.v1.utils import deep_update

from statsplotly import constants
from statsplotly._base import BaseModel
from statsplotly.constants import DEFAULT_HOVERMODE
from statsplotly.plot_objects.layout._axis import BaseAxis, ColorAxis, XYAxis
from statsplotly.plot_specifiers.data import DataDimension
from statsplotly.plot_specifiers.layout import (
    AxesSpecifier,
    BarMode,
    HistogramBarMode,
    PlotAxis,
)

logger = logging.getLogger(__name__)


DATA_TO_AXIS_MAP = {DataDimension.X: PlotAxis.XAXIS, DataDimension.Y: PlotAxis.YAXIS}


class _BaseLayout(BaseModel):
    autosize: bool | None = None
    hovermode: str = DEFAULT_HOVERMODE
    title: str | None = None
    height: int | None = None
    width: int | None = None
    showlegend: bool | None = None


class _XYLayout(_BaseLayout):
    xaxis: XYAxis
    yaxis: XYAxis

    @classmethod
    def build_xy_layout(cls, axes_specifier: AxesSpecifier) -> _XYLayout:
        xaxis_layout = XYAxis(
            title=axes_specifier.legend.xaxis_title,
            range=axes_specifier.xaxis_range,
        )
        yaxis_layout = XYAxis(
            title=axes_specifier.legend.yaxis_title,
            range=axes_specifier.yaxis_range,
            scaleanchor=axes_specifier.scaleanchor,
            scaleratio=axes_specifier.scaleratio,
        )
        return cls(
            title=axes_specifier.legend.figure_title,
            xaxis=xaxis_layout,
            yaxis=yaxis_layout,
            height=axes_specifier.height,
            width=axes_specifier.width,
        )


class _XYColorAxisLayout(_XYLayout):
    coloraxis: ColorAxis

    @classmethod
    def build_coloraxis_layout(
        cls, axes_specifier: AxesSpecifier, coloraxis: ColorAxis
    ) -> _XYColorAxisLayout:
        if coloraxis.colorbar is not None:
            coloraxis.colorbar.update(
                {
                    "x": constants.COLORBAR_XOFFSET,
                    "len": coloraxis.colorbar.get("len", 1) * constants.COLORBAR_REDUCTION_FACTOR,
                }
            )
            if coloraxis.colorbar.get("y") is None:
                coloraxis.colorbar.update({"y": 0, "yanchor": "bottom"})

        return cls(
            **_XYLayout.build_xy_layout(axes_specifier=axes_specifier).model_dump(),
            coloraxis=coloraxis,
        )


class SceneLayout(_BaseLayout):
    scene: dict[str, Any]
    coloraxis: ColorAxis

    @classmethod
    def build_layout(cls, axes_specifier: AxesSpecifier, coloraxis: ColorAxis) -> SceneLayout:
        scene = {
            "xaxis": BaseAxis(
                title=axes_specifier.legend.xaxis_title,
                range=axes_specifier.xaxis_range,
            ),
            "yaxis": BaseAxis(
                title=axes_specifier.legend.yaxis_title,
                range=axes_specifier.yaxis_range,
            ),
            "zaxis": BaseAxis(
                title=axes_specifier.legend.zaxis_title,
                range=axes_specifier.zaxis_range,
            ),
        }
        if coloraxis.colorbar is not None:
            coloraxis.colorbar.update(
                {
                    "x": constants.COLORBAR_XOFFSET,
                    "len": coloraxis.colorbar.get("len", 1) * constants.COLORBAR_REDUCTION_FACTOR,
                }
            )
            if coloraxis.colorbar.get("y") is None:
                coloraxis.colorbar.update({"y": 0, "yanchor": "bottom"})

        return cls(
            title=axes_specifier.legend.figure_title,
            scene=scene,
            height=axes_specifier.height,
            width=axes_specifier.width,
            coloraxis=coloraxis,
        )


class HeatmapLayout(_XYLayout):
    coloraxis: ColorAxis

    @classmethod
    def update_axis_layout(cls, axis_layout: XYAxis) -> XYAxis:
        axis_layout_dict = axis_layout.model_dump()
        update_keys: dict[str, Any] = {
            "showgrid": False,
            "zeroline": False,
            "constrain": "domain",
        }
        axis_layout_dict.update(update_keys)

        return XYAxis.model_validate(axis_layout_dict)

    @classmethod
    def update_yaxis_layout(cls, yaxis_layout: XYAxis) -> XYAxis:
        yaxis_layout_dict = cls.update_axis_layout(yaxis_layout).model_dump()

        update_keys: dict[str, Any] = {
            "autorange": "reversed",
            "range": (
                yaxis_layout_dict.get("range")[::-1]  # type: ignore
                if yaxis_layout_dict.get("range") is not None
                else None
            ),
            "constrain": "domain",
        }
        yaxis_layout_dict.update(update_keys)

        return XYAxis.model_validate(yaxis_layout_dict)

    @classmethod
    def build_layout(cls, axes_specifier: AxesSpecifier, coloraxis: ColorAxis) -> HeatmapLayout:
        base_layout = _XYLayout.build_xy_layout(axes_specifier=axes_specifier)

        heatmap_layout = deep_update(
            base_layout.model_dump(),
            {
                "xaxis": cls.update_axis_layout(
                    base_layout.xaxis,
                ).model_dump(),
                "yaxis": cls.update_yaxis_layout(
                    base_layout.yaxis,
                ).model_dump(),
            },
        )
        if coloraxis.colorbar is not None:
            coloraxis.colorbar.update({"title": axes_specifier.legend.zaxis_title})
        heatmap_layout.update({"coloraxis": coloraxis})

        return cls.model_validate(heatmap_layout)


class CategoricalLayout(_XYColorAxisLayout):
    boxmode: str = "group"
    violinmode: str = "group"

    @classmethod
    def set_array_tick_mode(
        cls, axis_layout: XYAxis, categorical_values_map: dict[str, Any]
    ) -> XYAxis:
        updated_dict = dict.fromkeys(["tickmode", "tickvals", "ticktext"], None)
        updated_dict["tickmode"] = "array"
        updated_dict["tickvals"] = [k + 1 for k in range(len(categorical_values_map))]
        updated_dict["ticktext"] = list(categorical_values_map.keys())

        return XYAxis.model_validate(deep_update(axis_layout.model_dump(), updated_dict))

    @classmethod
    def build_layout(
        cls,
        axes_specifier: AxesSpecifier,
        categorical_values_map: dict[DataDimension, dict[str, Any]] | None,
        coloraxis: ColorAxis,
    ) -> CategoricalLayout:
        base_layout = _XYColorAxisLayout.build_coloraxis_layout(
            axes_specifier=axes_specifier, coloraxis=coloraxis
        )

        if categorical_values_map is not None:
            for dimension, values_map in categorical_values_map.items():
                axis_ref = DATA_TO_AXIS_MAP[dimension]
                axis_layout = cls.set_array_tick_mode(
                    axis_layout=getattr(base_layout, axis_ref),
                    categorical_values_map=values_map,
                )
                return cls.model_validate(
                    deep_update(base_layout.model_dump(), {axis_ref: axis_layout.model_dump()})
                )
        return cls.model_validate(base_layout.model_dump())


class ScatterLayout(_XYColorAxisLayout):

    @classmethod
    def build_layout(
        cls,
        axes_specifier: AxesSpecifier,
        coloraxis: ColorAxis,
    ) -> ScatterLayout:

        return cls.model_validate(
            super().build_coloraxis_layout(axes_specifier=axes_specifier, coloraxis=coloraxis)
        )


class BarLayout(_XYColorAxisLayout):
    barmode: BarMode | None = None

    @classmethod
    def build_layout(
        cls,
        axes_specifier: AxesSpecifier,
        coloraxis: ColorAxis,
        barmode: str | None,
    ) -> BarLayout:

        return cls(
            **_XYColorAxisLayout.build_coloraxis_layout(
                axes_specifier=axes_specifier, coloraxis=coloraxis
            ).model_dump(),
            barmode=barmode,
        )


class HistogramLayout(_XYLayout):
    barmode: HistogramBarMode

    @field_validator("barmode", mode="before")
    def validate_histogram_barmode(cls, value: str | None) -> HistogramBarMode:
        if value is None:
            return HistogramBarMode.OVERLAY

        return HistogramBarMode(value)

    @classmethod
    def build_layout(cls, axes_specifier: AxesSpecifier, barmode: str | None) -> HistogramLayout:

        return cls(
            **_XYLayout.build_xy_layout(axes_specifier=axes_specifier).model_dump(), barmode=barmode
        )
