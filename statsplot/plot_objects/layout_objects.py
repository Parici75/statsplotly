import logging
from typing import Any, Dict, List

from pydantic import validator
from pydantic.utils import deep_update

from statsplot.constants import AXIS_TITLEFONT, TICKFONT
from statsplot.exceptions import StatsPlotInvalidArgumentError
from statsplot.plot_specifiers.data import BaseModel
from statsplot.plot_specifiers.layout import AxesSpecifier, BarMode

logger = logging.getLogger(__name__)


class ColorAxis(BaseModel):
    cmin: float | None = None
    cmax: float | None = None
    colorbar: Dict[str, Any] | None
    colorscale: str | List | None
    showscale: bool | None


class BaseAxisLayout(BaseModel):
    """Compatible properties with 2D and 3D (Scene) Layout."""

    autorange: bool | str | None
    title: str | None
    titlefont: Dict[str, Any] = AXIS_TITLEFONT
    tickfont: Dict[str, Any] = TICKFONT
    range: List | None
    type: str | None
    showgrid: bool | None
    tickmode: str | None
    tickvals: List | None
    ticktext: List | None
    zeroline: bool | None


class AxisLayout(BaseAxisLayout):
    automargin: bool = True
    scaleanchor: str | None
    scaleratio: float | None


class BaseLayout(BaseModel):
    autosize: bool | None
    hovermode: str = "closest"
    title: str | None
    height: int | None
    width: int | None
    showlegend: bool | None


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
    scene: Dict[str, Any]
    coloraxis: Dict[str, Any] | None

    @classmethod
    def build_layout(
        cls, axes_specifier: AxesSpecifier, coloraxis: ColorAxis
    ) -> "SceneLayout":
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
        axis_layout_dict = axis_layout.dict()
        update_keys: Dict[str, Any] = {
            "showgrid": False,
            "zeroline": False,
        }
        axis_layout_dict.update(update_keys)

        return AxisLayout.parse_obj(axis_layout_dict)

    @classmethod
    def update_yaxis_layout(cls, yaxis_layout: AxisLayout) -> AxisLayout:
        yaxis_layout_dict = cls.update_axis_layout(yaxis_layout).dict()

        update_keys: Dict[str, Any] = {
            "autorange": "reversed",
            "range": yaxis_layout_dict.get("range")[::-1]  # type: ignore
            if yaxis_layout_dict.get("range") is not None
            else None,
        }
        yaxis_layout_dict.update(update_keys)

        return AxisLayout.parse_obj(yaxis_layout_dict)

    @classmethod
    def build_layout(
        cls, axes_specifier: AxesSpecifier, coloraxis: ColorAxis
    ) -> "HeatmapLayout":
        base_layout = XYLayout.build_xy_layout(axes_specifier=axes_specifier)

        heatmap_layout = deep_update(
            base_layout.dict(),
            {
                "xaxis": cls.update_axis_layout(
                    base_layout.xaxis,
                ).dict(),
                "yaxis": cls.update_yaxis_layout(
                    base_layout.yaxis,
                ).dict(),
            },
        )
        heatmap_layout.update({"coloraxis": coloraxis})

        return cls.parse_obj(heatmap_layout)


class CategoricalLayout(XYLayout):
    boxmode: str = "group"
    violinmode: str = "group"

    @classmethod
    def set_array_tick_mode(
        cls, axis_layout: AxisLayout, x_values_map: Dict[str, Any]
    ) -> AxisLayout:
        updated_dict = dict.fromkeys(
            ["tickmode", "tickvals", "ticktext"], None
        )
        updated_dict["tickmode"] = "array"
        updated_dict["tickvals"] = [k + 1 for k in range(len(x_values_map))]
        updated_dict["ticktext"] = list(x_values_map.keys())

        return AxisLayout.parse_obj(
            deep_update(axis_layout.dict(), updated_dict)
        )

    @classmethod
    def build_layout(
        cls, axes_specifier: AxesSpecifier, x_values_map: Dict[str, Any] | None
    ) -> "CategoricalLayout":
        base_layout = XYLayout.build_xy_layout(axes_specifier=axes_specifier)

        if x_values_map is not None:
            xaxis_layout = cls.set_array_tick_mode(
                axis_layout=base_layout.xaxis, x_values_map=x_values_map
            )
            return cls.parse_obj(
                deep_update(base_layout.dict(), {"xaxis": xaxis_layout.dict()})
            )

        return cls.parse_obj(base_layout)


class ScatterLayout(XYLayout):
    coloraxis: Dict[str, Any]

    @classmethod
    def build_layout(
        cls,
        axes_specifier: AxesSpecifier,
        coloraxis: ColorAxis,
    ) -> "ScatterLayout":
        base_layout = XYLayout.build_xy_layout(axes_specifier=axes_specifier)
        return cls(**base_layout.dict(), coloraxis=coloraxis)


class BarLayout(XYLayout):
    barmode: BarMode | None
    coloraxis: Dict[str, Any]

    @validator("barmode", pre=True)
    def check_barmode(cls, value: str | None) -> BarMode | None:
        try:
            return BarMode(value) if value is not None else None
        except ValueError as exc:
            raise StatsPlotInvalidArgumentError(
                value, BarMode  # type: ignore
            ) from exc

    @classmethod
    def build_layout(
        cls,
        axes_specifier: AxesSpecifier,
        coloraxis: ColorAxis,
        barmode: str | None,
    ) -> "BarLayout":
        scatter_layout = XYLayout.build_xy_layout(
            axes_specifier=axes_specifier
        )
        return cls(
            **scatter_layout.dict(), coloraxis=coloraxis, barmode=barmode
        )


class HistogramLayout(XYLayout):
    barmode: BarMode | None

    @classmethod
    def build_layout(
        cls, axes_specifier: AxesSpecifier, barmode: str | None
    ) -> "HistogramLayout":
        base_layout = XYLayout.build_xy_layout(axes_specifier=axes_specifier)

        return cls(**base_layout.dict(), barmode=barmode)
