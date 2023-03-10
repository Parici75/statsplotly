import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_object_dtype,
)
from pydantic import validator

from pymodules.pandas_utils import unique_non_null
from statsplot import constants
from statsplot.exceptions import StatsPlotSpecificationError
from statsplot.plot_objects.layout_objects import ColorAxis
from statsplot.plot_specifiers.data import BaseModel
from statsplot.plot_specifiers.layout import ColoraxisReference
from statsplot.utils.colors_utils import (
    ColorSystem,
    compute_colorscale,
    get_rgb_discrete_array,
)
from statsplot.utils.layout_utils import smart_legend

logger = logging.getLogger(__name__)


class ColorSpecifier(BaseModel):
    coloraxis_reference: ColoraxisReference | None
    logscale: float | None
    color_palette: str | List[str] | None = constants.DEFAULT_COLOR_PALETTE
    color_limits: List | None
    colorbar: bool | None
    opacity: float | None

    @validator("logscale")
    def check_logscale(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise StatsPlotSpecificationError(
                "Logscale base must be greater than 0"
            )
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

    def build_colorbar(
        self, color_values: pd.Series | None
    ) -> Dict[str, Any] | None:
        if color_values is None:
            return None

        if is_bool_dtype(color_values.dtype) or is_object_dtype(
            color_values.dtype
        ):
            unique_color_values = np.sort(color_values.astype(int).unique())

            ticklimits = np.linspace(
                *unique_color_values[[0, -1]], len(unique_color_values) + 1  # type: ignore
            )
            return dict(
                title=smart_legend(color_values.name),
                len=0.5,
                yanchor="top",
                tickmode="array",
                tickvals=(ticklimits[:-1] + ticklimits[1:]) / 2,
                ticktext=unique_color_values,
            )

        return dict(
            title=smart_legend(color_values.name),
            len=0.5,
            yanchor="top",
            tickmode="auto",
        )

    def build_colorscale(
        self, color_data: pd.Series | None
    ) -> str | List | None:
        if color_data is None:
            return None

        # Select the appropriate color system
        if is_bool_dtype(color_data.dtype) or is_object_dtype(
            color_data.dtype
        ):
            try:
                color_data.astype(int)
            except ValueError:
                logger.debug(
                    f"{color_data.name} values are not numeric, assume direct color specification"
                )
                return None

            n_colors = len(unique_non_null(color_data))
            color_system = ColorSystem.DISCRETE
        else:
            n_colors = constants.N_COLORSCALE_COLORS
            color_system = ColorSystem.LINEAR

        if self.logscale is not None:
            if color_system is ColorSystem.DISCRETE:
                raise ValueError(
                    f"{ColorSystem.LOGARITHMIC.value} color system is not compatible with {ColorSystem.DISCRETE.value} colormapping"
                )
            color_system = ColorSystem.LOGARITHMIC

        colorscale = compute_colorscale(
            n_colors,
            color_system=color_system,
            logscale=self.logscale,
            color_palette=self.color_palette,
        )

        return colorscale

    def build_coloraxis(
        self, color_data: pd.Series | None, shared: bool = False
    ) -> ColorAxis:
        if shared and color_data is not None:
            cmin = color_data.min() if self.cmin is None else self.cmin
            cmax = color_data.max() if self.cmax is None else self.cmin
        else:
            cmin, cmax = self.cmin, self.cmax

        colorscale = self.build_colorscale(color_data)
        colorbar = (
            self.build_colorbar(color_data) if colorscale is not None else None
        )

        return ColorAxis(
            cmin=cmin,
            cmax=cmax,
            colorscale=colorscale,
            colorbar=colorbar,
            showscale=self.colorbar if colorscale is not None else None,
        )

    def get_color_hues(self, n_colors: int) -> List[str]:
        return get_rgb_discrete_array(
            color_palette=self.color_palette, n_colors=n_colors
        )
