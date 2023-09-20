import logging

import numpy as np
import pandas as pd
import seaborn as sns

from statsplotly import constants
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.utils.colors_utils import (
    ColorSystem,
    cmap_to_array,
    compute_colorscale,
    to_rgb,
)

EXAMPLE_VALID_COLOR_DATA = pd.Series(np.random.randint(1, 100, 100), name="color_data")
EXAMPLE_DIRECT_COLOR_ARRAY = pd.Series(["green", "red", "blue"], name="colors")

logging.getLogger().setLevel(logging.DEBUG)


class TestColorSpecifier:
    def test_builtin_colorscale(self):
        color_axis = ColorSpecifier(color_palette="viridis").build_coloraxis(
            color_data=EXAMPLE_VALID_COLOR_DATA
        )
        assert color_axis.colorscale == "viridis"

    def test_continuous_colorscale(self):
        color_axis = ColorSpecifier().build_coloraxis(color_data=EXAMPLE_VALID_COLOR_DATA)
        assert [color_tuple[1] for color_tuple in color_axis.colorscale] == to_rgb(
            cmap_to_array(
                constants.N_COLORSCALE_COLORS,
                sns.color_palette(
                    palette=constants.SEABORN_DEFAULT_CONTINUOUS_COLOR_PALETTE,
                    as_cmap=True,
                ),
            )
        )

    def test_direct_color_array(self, caplog):
        color_axis = ColorSpecifier().build_coloraxis(color_data=EXAMPLE_DIRECT_COLOR_ARRAY)
        assert color_axis.colorscale is None and color_axis.colorbar is None
        assert (
            f"{EXAMPLE_DIRECT_COLOR_ARRAY.name} values are not numeric, assuming direct color"
            " specification"
            in caplog.text
        )

    def test_builtin_colormap(self):
        color_axis = ColorSpecifier(color_palette="Blues").build_coloraxis(
            color_data=EXAMPLE_VALID_COLOR_DATA
        )
        assert color_axis.colorscale == "Blues"

    def test_listed_colormap(self):
        color_axis = ColorSpecifier(color_palette=["#05513c", "#c2efdf"]).build_coloraxis(
            color_data=EXAMPLE_VALID_COLOR_DATA
        )
        assert color_axis.colorscale == compute_colorscale(
            n_colors=constants.N_COLORSCALE_COLORS,
            color_system=ColorSystem.LINEAR,
            color_palette=["#05513c", "#c2efdf"],
        )

    def test_logarithmic_scale(self):
        color_axis = ColorSpecifier(
            logscale=10, color_palette=["#05513c", "#c2efdf"]
        ).build_coloraxis(color_data=EXAMPLE_VALID_COLOR_DATA)
        assert color_axis.colorscale == compute_colorscale(
            n_colors=constants.N_COLORSCALE_COLORS,
            color_system=ColorSystem.LOGARITHMIC,
            color_palette=["#05513c", "#c2efdf"],
        )

    def test_discrete_colormap(self):
        color_axis = ColorSpecifier(color_palette="Set3").build_coloraxis(
            color_data=EXAMPLE_VALID_COLOR_DATA.astype(str)
        )
        # Verify colormap indices are repeated every two values
        assert np.diff(np.array([value[0] for value in color_axis.colorscale]))[1::2].sum() == 0

    def test_get_color_hues(self):
        color_hues = ColorSpecifier(color_palette="tab10").get_color_hues(3)
        assert len(color_hues) == 3
