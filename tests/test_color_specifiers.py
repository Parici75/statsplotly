import logging

import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from statsplotly import constants
from statsplotly.exceptions import StatsPlotSpecificationError
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.plot_specifiers.color._utils import (
    ColorSystem,
    cmap_to_array,
    compute_colorscale,
    to_rgb_string,
)

EXAMPLE_VALID_INT_COLOR_DATA = pd.Series(np.random.randint(1, 100, 100), name="color_data")
EXAMPLE_VALID_FLOAT_COLOR_DATA = pd.Series(np.random.rand(100), name="color_data")
EXAMPLE_DIRECT_COLOR_ARRAY = pd.Series(["green", "red", "blue"], name="colors")
EXAMPLE_VALID_DATETIME_COLOR_DATA = pd.Series(
    pd.to_datetime(("2020-01-01", "2020-01-02", "2020-02-03")), name="color_data"
)
EXAMPLE_INT_MAPPED_COLOR_ARRAY = pd.Series(["0", "1", "2"], name="color_ids")


logging.getLogger().setLevel(logging.DEBUG)


class TestColorSpecifier:

    def test_color_specifier(self):
        color_specifier = ColorSpecifier.build_from_color_data(EXAMPLE_VALID_FLOAT_COLOR_DATA)
        assert color_specifier.colormap is None

        color_specifier = ColorSpecifier.build_from_color_data(
            EXAMPLE_VALID_INT_COLOR_DATA.astype(str)
        )
        color_specifier.colormap = dict(
            zip(
                EXAMPLE_VALID_INT_COLOR_DATA.dropna().unique(),
                range(len(EXAMPLE_VALID_INT_COLOR_DATA.dropna().unique())),
                strict=True,
            )
        )

    def test_builtin_plotly_colorscale(self):
        coloraxis = ColorSpecifier(color_palette="viridis").build_coloraxis(
            color_data=EXAMPLE_VALID_INT_COLOR_DATA
        )
        assert coloraxis.colorscale == "viridis"

    def test_builtin_mpl_colorscale(self, caplog):
        coloraxis = ColorSpecifier(color_palette="winter").build_coloraxis(
            color_data=EXAMPLE_VALID_INT_COLOR_DATA
        )
        assert [color_tuple[1] for color_tuple in coloraxis.colorscale] == [
            to_rgb_string(color)
            for color in cmap_to_array(
                constants.N_COLORSCALE_COLORS,
                "winter",
            )
        ]
        assert "Plotly error processing winter colormap" in caplog.text

    def test_listed_colormap(self):
        coloraxis = ColorSpecifier(color_palette=["#05513c", "#c2efdf"]).build_coloraxis(
            color_data=EXAMPLE_VALID_INT_COLOR_DATA
        )
        assert coloraxis.colorscale == compute_colorscale(
            n_colors=constants.N_COLORSCALE_COLORS,
            color_system=ColorSystem.LINEAR,
            color_palette=["#05513c", "#c2efdf"],
        )

    def test_seaborn_continuous_colorscale(self):
        coloraxis = ColorSpecifier().build_coloraxis(color_data=EXAMPLE_VALID_INT_COLOR_DATA)
        assert [color_tuple[1] for color_tuple in coloraxis.colorscale] == [
            to_rgb_string(color)
            for color in cmap_to_array(
                constants.N_COLORSCALE_COLORS,
                sns.color_palette(
                    palette=constants.SEABORN_DEFAULT_CONTINUOUS_COLOR_PALETTE,
                    as_cmap=True,
                ),
            )
        ]
        assert coloraxis.colorscale is not None

    def test_format_color_data(self, caplog):
        # String
        single_string_color_data = ColorSpecifier().format_color_data(color_data="blue")
        assert single_string_color_data == "blue"

        # Direct
        direct_color_data = ColorSpecifier().format_color_data(
            color_data=EXAMPLE_DIRECT_COLOR_ARRAY
        )
        assert all(direct_color_data == EXAMPLE_DIRECT_COLOR_ARRAY)
        assert (
            f"{EXAMPLE_DIRECT_COLOR_ARRAY.name} values are all color-like, statsplotly will assume direct color specification"
            in caplog.text
        )

        # Integer
        integer_color_data = ColorSpecifier().format_color_data(EXAMPLE_VALID_INT_COLOR_DATA)
        assert all(integer_color_data == EXAMPLE_VALID_INT_COLOR_DATA)

        # Float
        float_color_data = ColorSpecifier().format_color_data(
            color_data=EXAMPLE_VALID_FLOAT_COLOR_DATA
        )
        assert all(float_color_data == EXAMPLE_VALID_FLOAT_COLOR_DATA)

        # Datetime
        timestamp_color_data = ColorSpecifier().format_color_data(
            color_data=EXAMPLE_VALID_DATETIME_COLOR_DATA
        )
        expected_output = [1.577837e09, 1.577923e09, 1.580688e09]
        assert np.allclose(
            timestamp_color_data, [1.577837e09, 1.577923e09, 1.580688e09]
        ), f"Expected {expected_output} but got {result}"

        # Mapping
        mapped_color_data = ColorSpecifier.build_from_color_data(
            EXAMPLE_INT_MAPPED_COLOR_ARRAY
        ).format_color_data(color_data=EXAMPLE_INT_MAPPED_COLOR_ARRAY)
        assert all(
            mapped_color_data
            == EXAMPLE_INT_MAPPED_COLOR_ARRAY.map(
                dict(
                    zip(
                        EXAMPLE_INT_MAPPED_COLOR_ARRAY.dropna().unique(),
                        range(len(EXAMPLE_INT_MAPPED_COLOR_ARRAY.dropna().unique())),
                        strict=True,
                    )
                )
            )
        )
        assert (
            f"{EXAMPLE_INT_MAPPED_COLOR_ARRAY.name} values of type='object' are not continuous type, statsplotly will map it to colormap"
            in caplog.text
        )

        with pytest.raises(StatsPlotSpecificationError) as excinfo:
            ColorSpecifier().format_color_data(color_data=EXAMPLE_INT_MAPPED_COLOR_ARRAY)
        assert (
            f"No colormap attribute to map discrete data onto, check {ColorSpecifier.__name__} instantiation"
            in str(excinfo.value)
        )

    def test_direct_color_array(self, caplog):
        coloraxis = ColorSpecifier().build_coloraxis(color_data=EXAMPLE_DIRECT_COLOR_ARRAY)
        assert coloraxis.colorscale is None and coloraxis.colorbar is None

    def test_logarithmic_scale(self):
        coloraxis = ColorSpecifier(
            logscale=10, color_palette=["#05513c", "#c2efdf"]
        ).build_coloraxis(color_data=EXAMPLE_VALID_INT_COLOR_DATA)
        assert coloraxis.colorscale == compute_colorscale(
            n_colors=constants.N_COLORSCALE_COLORS,
            color_system=ColorSystem.LOGARITHMIC,
            color_palette=["#05513c", "#c2efdf"],
        )

    def test_discrete_integer_coloraxis(self):
        coloraxis = ColorSpecifier.build_from_color_data(
            color_data=EXAMPLE_VALID_INT_COLOR_DATA.astype(str), color_palette="Set3"
        ).build_coloraxis(color_data=EXAMPLE_VALID_INT_COLOR_DATA.astype(str))
        # Verify colormap indices are repeated every two values
        assert np.diff(np.array([value[0] for value in coloraxis.colorscale]))[1::2].sum() == 0

    def test_get_color_hues(self):
        color_hues = ColorSpecifier(color_palette="tab10").get_color_hues(3)
        assert len(color_hues) == 3
