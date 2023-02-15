import numpy as np
import pandas as pd

from statsplot import constants
from statsplot.plot_specifiers.color import ColorSpecifier
from statsplot.utils.colors_utils import compute_colorscale, ColorSystem

EXAMPLE_COLOR_DATA = pd.Series(np.random.randint(1, 100, 100), name="color_data")


class TestColorSpecifier:
    def test_color_specifier(self):
        color_axis = ColorSpecifier().build_coloraxis(
            color_data=EXAMPLE_COLOR_DATA)
        assert color_axis.colorscale == constants.DEFAULT_COLOR_PALETTE

    def test_builtin_colormap(self):
        color_axis = ColorSpecifier(color_palette="Blues").build_coloraxis(
            color_data=EXAMPLE_COLOR_DATA)
        assert color_axis.colorscale == "Blues"

    def test_listed_colormap(self):
        color_axis = ColorSpecifier(
            color_palette=["#05513c", "#c2efdf"]).build_coloraxis(
            color_data=EXAMPLE_COLOR_DATA)
        assert color_axis.colorscale == compute_colorscale(
            n_colors=constants.N_COLORSCALE_COLORS,
            color_system=ColorSystem.LINEAR,
            color_palette=["#05513c", "#c2efdf"])

    def test_logarithmic_scale(self):
        color_axis = ColorSpecifier(
            logscale=10,
            color_palette=["#05513c", "#c2efdf"]).build_coloraxis(
            color_data=EXAMPLE_COLOR_DATA)
        assert color_axis.colorscale == compute_colorscale(
            n_colors=constants.N_COLORSCALE_COLORS,
            color_system=ColorSystem.LOGARITHMIC,
            color_palette=["#05513c", "#c2efdf"])

    def test_discrete_colormap(self):
        color_axis = ColorSpecifier(color_palette="Set3").build_coloraxis(
            color_data=EXAMPLE_COLOR_DATA.astype(str))
        # Verify colormap indices are repeated every two values
        assert np.diff(np.array([value[0] for value in color_axis.colorscale]))[1::2].sum() == 0

    def test_get_color_hues(self):
        color_hues = ColorSpecifier(color_palette="tab10").get_color_hues(
            3)
        assert len(color_hues) == 3


