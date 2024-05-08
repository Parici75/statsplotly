from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import is_color_like
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_object_dtype,
)
from pydantic import ValidationInfo, field_validator

from statsplotly import constants
from statsplotly._base import BaseModel
from statsplotly.exceptions import StatsPlotSpecificationError
from statsplotly.plot_objects.layout import ColorAxis
from statsplotly.plot_specifiers.common import smart_legend
from statsplotly.plot_specifiers.layout import BarMode, ColoraxisReference

from ._utils import ColorSystem, compute_colorscale, rgb_string_array_from_colormap

logger = logging.getLogger(__name__)


class ColorSpecifier(BaseModel):
    barmode: BarMode | None = None
    coloraxis_reference: ColoraxisReference | None = None
    colormap: dict[str | np.datetime64 | bool, Any] | None = None
    logscale: float | None = None
    color_palette: str | list[str] | None = None
    color_limits: list[float] | None = None
    colorbar: bool | None = None
    opacity: float | None = None

    @field_validator("opacity", mode="before")
    def check_opacity(cls, value: str | float | None) -> float | None:
        if isinstance(value, str):
            logger.debug("Opacity argument is string, hence defined for marker level")
            return None
        return value

    @field_validator("logscale")
    def check_logscale(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise StatsPlotSpecificationError("Logscale base must be greater than 0")
        return value

    @property
    def zmin(self) -> float | None:
        return self.color_limits[0] if self.color_limits is not None else None

    @property
    def zmax(self) -> float | None:
        return self.color_limits[-1] if self.color_limits is not None else None

    @property
    def cmin(self) -> float | None:
        return self.zmin

    @property
    def cmax(self) -> float | None:
        return self.zmax

    @staticmethod
    def _check_is_discrete_color_data_type(color_data: pd.Series) -> bool:
        return is_bool_dtype(color_data) or is_object_dtype(color_data)

    @staticmethod
    def _check_is_direct_color_specification(color_data: pd.Series) -> bool:
        if ColorSpecifier._check_is_discrete_color_data_type(color_data):
            return all(color_data.replace("0|1", "", regex=True).map(is_color_like))

        return False

    @staticmethod
    def _check_is_datetime_color_data_type(color_data: pd.Series) -> bool:
        return is_datetime64_any_dtype(color_data)

    @staticmethod
    def convert_datetime_to_timestamp(x: np.datetime64) -> float:
        try:
            return x.timestamp()  # type: ignore
        except ValueError:
            return np.nan

    @classmethod
    def _register_colormap(
        cls, color_data: pd.Series
    ) -> dict[str | np.datetime64 | bool, Any] | None:
        if color_data is None:
            return None

        if cls._check_is_direct_color_specification(color_data):
            logger.debug(
                f"{color_data.name} values are all color-like, statsplotly will assume direct color specification"
            )

        if cls._check_is_discrete_color_data_type(color_data):
            return dict(
                zip(
                    color_data.dropna().unique(),
                    np.arange(len(color_data.dropna().unique())),
                    strict=True,
                )
            )

        return None

    def format_color_data(self, color_data: str | pd.Series) -> pd.Series:
        if self._check_is_direct_color_specification(color_data):
            logger.debug(
                f"{color_data.name} values are all color-like, statsplotly will assume direct color specification"  # type: ignore
            )
            return color_data

        if self._check_is_datetime_color_data_type(color_data):
            return color_data.map(self.convert_datetime_to_timestamp)  # type: ignore

        if self._check_is_discrete_color_data_type(color_data):
            logger.debug(
                f"{color_data.name} values of type='{color_data.dtype}' are not continuous type, statsplotly will map it to colormap"  # type: ignore
            )
            if self.colormap is None:
                raise StatsPlotSpecificationError(
                    f"No colormap attribute to map discrete data onto, check {ColorSpecifier.__name__} instantiation"
                )
            return color_data.map(self.colormap)  # type: ignore

        return color_data

    def build_colorbar(self, color_values: pd.Series | None) -> dict[str, Any] | None:
        if color_values is None:
            return None

        colorbar_dict = {
            "title": smart_legend(color_values.name),
            "len": 1,
            "xanchor": "left",
            "yanchor": "middle",
            "tickmode": "auto",
        }

        if self._check_is_discrete_color_data_type(
            color_values
        ) or self._check_is_datetime_color_data_type(color_values):
            colorbar_dict.update({"tickmode": "array"})

            if self._check_is_discrete_color_data_type(color_values):
                if (colormap := self.colormap) is None:
                    raise StatsPlotSpecificationError(
                        f"No colormap defined, check {ColorSpecifier.__name__} instantiation"
                    )
                colormap_ticks = list(colormap.values())
                _tick_locations = np.linspace(
                    colormap_ticks[0], colormap_ticks[-1], len(colormap_ticks) + 1
                )
                tickvals = (_tick_locations[:-1] + _tick_locations[1:]) / 2
                ticktext = color_values.dropna().unique()

            elif self._check_is_datetime_color_data_type(color_values):
                tickvals = self.format_color_data(color_values).iloc[[0, -1]]
                ticktext = [
                    datum.strftime("%B %Y")
                    for datum in color_values.dropna().sort_values().iloc[[0, -1]]
                ]

            colorbar_dict.update(
                {
                    "tickvals": tickvals,
                    "ticktext": ticktext,
                }
            )

        return colorbar_dict

    def build_colorscale(
        self, color_data: pd.Series | None
    ) -> str | list[list[float | str]] | None:
        if color_data is None:
            return None

        if self._check_is_direct_color_specification(color_data):
            logger.debug(f"{color_data.name} data is all color-like, returning no colorscale")
            return None

        # Select the appropriate color system
        if self._check_is_discrete_color_data_type(color_data):
            _color_data = self.format_color_data(color_data)
            n_colors = _color_data.dropna().nunique()
            color_system = ColorSystem.DISCRETE
        else:
            n_colors = constants.N_COLORSCALE_COLORS
            color_system = ColorSystem.LINEAR

        if self.logscale is not None:
            if color_system is ColorSystem.DISCRETE:
                raise ValueError(
                    f"{ColorSystem.LOGARITHMIC.value} color system is not compatible with"
                    f" {ColorSystem.DISCRETE.value} colormapping"
                )
            color_system = ColorSystem.LOGARITHMIC

        colorscale = compute_colorscale(
            n_colors,
            color_system=color_system,
            logscale=self.logscale,
            color_palette=self.color_palette,
        )

        return colorscale

    def build_coloraxis(self, color_data: pd.Series | None, shared: bool = False) -> ColorAxis:
        if shared and color_data is not None:
            cmin = color_data.min() if self.cmin is None else self.cmin
            cmax = color_data.max() if self.cmax is None else self.cmin
        else:
            cmin, cmax = self.cmin, self.cmax

        colorscale = self.build_colorscale(color_data)
        colorbar = self.build_colorbar(color_data) if colorscale is not None else None

        return ColorAxis(
            cmin=cmin, cmax=cmax, colorscale=colorscale, colorbar=colorbar, showscale=self.colorbar
        )

    def get_color_hues(self, n_colors: int) -> list[str]:
        return rgb_string_array_from_colormap(color_palette=self.color_palette, n_colors=n_colors)

    @classmethod
    def build_from_color_data(
        cls,
        color_data: pd.Series,
        **kwargs: Any,
    ) -> ColorSpecifier:
        colormap = cls._register_colormap(color_data=color_data)

        return cls(
            colormap=colormap,
            **kwargs,
        )


class HistogramColorSpecifier(ColorSpecifier):
    opacity: float

    @field_validator("opacity", mode="before")
    def check_opacity(cls, value: str | float | None, info: ValidationInfo) -> float:
        if isinstance(value, str):
            raise StatsPlotSpecificationError("Histogram opacity must be float")

        if value is None:
            if info.data.get("barmode") is BarMode.OVERLAY:
                logger.info(
                    f"Setting up histogram opacity to {constants.DEFAULT_OVERLAID_HISTOGRAM_OPACITY} for `barmode={BarMode.OVERLAY.value}`"
                )
                return constants.DEFAULT_OVERLAID_HISTOGRAM_OPACITY

            return constants.DEFAULT_HISTOGRAM_OPACITY

        return value
