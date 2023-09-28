import datetime
import logging
from typing import Any

from pydantic import ValidationInfo, field_validator, model_validator
from pydantic.v1.utils import deep_update

from statsplotly.constants import AXIS_TITLEFONT, DEFAULT_HOVERMODE, TICKFONT
from statsplotly.exceptions import StatsPlotInvalidArgumentError
from statsplotly.plot_specifiers.data import BaseModel
from statsplotly.plot_specifiers.layout import AxesSpecifier, BarMode

logger = logging.getLogger(__name__)

axis_coordinates_type = float | datetime.datetime | str


class ColorAxis(BaseModel):
    cmin: float | None = None
    cmax: float | None = None
    colorbar: dict[str, Any] | None = None
    colorscale: str | list[list[float | str]] | None = None
    showscale: bool | None = None


class BaseAxisLayout(BaseModel):
    """Compatible properties with 2D and 3D (Scene) Layout."""

    title: str | None = None
    titlefont: dict[str, Any] = AXIS_TITLEFONT
    tickfont: dict[str, Any] = TICKFONT
    range: list[axis_coordinates_type] | None = None  # noqa: A003
    type: str | None = None  # noqa: A003
    autorange: bool | str | None = None
    showgrid: bool | None = None
    tickmode: str | None = None
    tickvals: list[axis_coordinates_type] | None = None
    ticktext: list[str] | None = None
    zeroline: bool | None = None

    @field_validator("autorange")
    def validate_autorange(
        cls, value: bool | str | None, info: ValidationInfo
    ) -> bool | str | None:
        if info.data.get("range") is not None:
            return False
        return value

    @model_validator(mode="after")  # type: ignore
    def validate_axis_consistency(cls, model: "BaseAxisLayout") -> "BaseAxisLayout":
        if model.range is not None and model.tickvals is not None:
            model.tickvals = model.range
            if model.ticktext is not None:
                try:
                    model.ticktext = [
                        model.ticktext[int(idx)] for idx in model.tickvals  # type: ignore
                    ]
                except TypeError:
                    logger.error("Can not adjust tick text labels to tick values")

        return model


class AxisLayout(BaseAxisLayout):
    automargin: bool = True
    scaleanchor: str | None = None
    scaleratio: float | None = None


class BaseLayout(BaseModel):
    autosize: bool | None = None
    hovermode: str = DEFAULT_HOVERMODE
    title: str | None = None
    height: int | None = None
    width: int | None = None
    showlegend: bool | None = None


class XYLayout(BaseLayout):
    xaxis: AxisLayout
    yaxis: AxisLayout

    @classmethod
    def build_xy_layout(cls, axes_specifier: AxesSpecifier) -> "XYLayout":
        xaxis_layout = AxisLayout(
            title=axes_specifier.legend.xaxis_title,
            range=axes_specifier.xaxis_range,
        )
        yaxis_layout = AxisLayout(
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


class SceneLayout(BaseLayout):
    scene: dict[str, Any]
    coloraxis: ColorAxis

    @classmethod
    def build_layout(cls, axes_specifier: AxesSpecifier, coloraxis: ColorAxis) -> "SceneLayout":
        scene = {
            "xaxis": BaseAxisLayout(
                title=axes_specifier.legend.xaxis_title,
                range=axes_specifier.xaxis_range,
            ),
            "yaxis": BaseAxisLayout(
                title=axes_specifier.legend.yaxis_title,
                range=axes_specifier.yaxis_range,
            ),
            "zaxis": BaseAxisLayout(
                title=axes_specifier.legend.zaxis_title,
                range=axes_specifier.zaxis_range,
            ),
        }

        return cls(
            title=axes_specifier.legend.figure_title,
            scene=scene,
            height=axes_specifier.height,
            width=axes_specifier.width,
            coloraxis=coloraxis,
        )


class HeatmapLayout(XYLayout):
    coloraxis: ColorAxis

    @classmethod
    def update_axis_layout(cls, axis_layout: AxisLayout) -> AxisLayout:
        axis_layout_dict = axis_layout.model_dump()
        update_keys: dict[str, Any] = {
            "showgrid": False,
            "zeroline": False,
        }
        axis_layout_dict.update(update_keys)

        return AxisLayout.model_validate(axis_layout_dict)

    @classmethod
    def update_yaxis_layout(cls, yaxis_layout: AxisLayout) -> AxisLayout:
        yaxis_layout_dict = cls.update_axis_layout(yaxis_layout).model_dump()

        update_keys: dict[str, Any] = {
            "autorange": "reversed",
            "range": (
                yaxis_layout_dict.get("range")[::-1]  # type: ignore
                if yaxis_layout_dict.get("range") is not None
                else None
            ),
        }
        yaxis_layout_dict.update(update_keys)

        return AxisLayout.model_validate(yaxis_layout_dict)

    @classmethod
    def build_layout(cls, axes_specifier: AxesSpecifier, coloraxis: ColorAxis) -> "HeatmapLayout":
        base_layout = XYLayout.build_xy_layout(axes_specifier=axes_specifier)

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
        heatmap_layout.update({"coloraxis": coloraxis})

        return cls.model_validate(heatmap_layout)


class CategoricalLayout(XYLayout):
    boxmode: str = "group"
    violinmode: str = "group"

    @classmethod
    def set_array_tick_mode(
        cls, axis_layout: AxisLayout, x_values_map: dict[str, Any]
    ) -> AxisLayout:
        updated_dict = dict.fromkeys(["tickmode", "tickvals", "ticktext"], None)
        updated_dict["tickmode"] = "array"
        updated_dict["tickvals"] = [k + 1 for k in range(len(x_values_map))]
        updated_dict["ticktext"] = list(x_values_map.keys())

        return AxisLayout.model_validate(deep_update(axis_layout.model_dump(), updated_dict))

    @classmethod
    def build_layout(
        cls, axes_specifier: AxesSpecifier, x_values_map: dict[str, Any] | None
    ) -> "CategoricalLayout":
        base_layout = XYLayout.build_xy_layout(axes_specifier=axes_specifier)

        if x_values_map is not None:
            xaxis_layout = cls.set_array_tick_mode(
                axis_layout=base_layout.xaxis, x_values_map=x_values_map
            )
            return cls.model_validate(
                deep_update(base_layout.model_dump(), {"xaxis": xaxis_layout.model_dump()})
            )

        return cls.model_validate(base_layout.model_dump())


class ScatterLayout(XYLayout):
    coloraxis: ColorAxis

    @classmethod
    def build_layout(
        cls,
        axes_specifier: AxesSpecifier,
        coloraxis: ColorAxis,
    ) -> "ScatterLayout":
        base_layout = XYLayout.build_xy_layout(axes_specifier=axes_specifier)
        return cls(**base_layout.model_dump(), coloraxis=coloraxis)


class BarLayout(XYLayout):
    barmode: BarMode | None = None
    coloraxis: ColorAxis

    @field_validator("barmode", mode="before")
    def check_barmode(cls, value: str | None) -> BarMode | None:
        try:
            return BarMode(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(value, BarMode) from exc  # type: ignore

    @classmethod
    def build_layout(
        cls,
        axes_specifier: AxesSpecifier,
        coloraxis: ColorAxis,
        barmode: str | None,
    ) -> "BarLayout":
        scatter_layout = XYLayout.build_xy_layout(axes_specifier=axes_specifier)
        return cls(**scatter_layout.model_dump(), coloraxis=coloraxis, barmode=barmode)


class HistogramLayout(XYLayout):
    barmode: BarMode | None = None

    @classmethod
    def build_layout(cls, axes_specifier: AxesSpecifier, barmode: str | None) -> "HistogramLayout":
        base_layout = XYLayout.build_xy_layout(axes_specifier=axes_specifier)

        return cls(**base_layout.model_dump(), barmode=barmode)
