import logging
from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic.v1.utils import deep_update

from statsplotly import constants
from statsplotly.plot_specifiers.color import ColorSpecifier
from statsplotly.plot_specifiers.data import (
    TRACE_DIMENSION_MAP,
    AggregationTraceData,
    BaseModel,
    DataDimension,
    HistogramNormType,
    RegressionType,
    TraceData,
)
from statsplotly.plot_specifiers.trace import (
    HistogramSpecifier,
    JointplotSpecifier,
    JointplotType,
    TraceMode,
)
from statsplotly.utils.color_utils import set_rgb_alpha
from statsplotly.utils.stats_utils import (
    affine_func,
    exponential_regress,
    inverse_func,
    kde_1d,
    kde_2d,
    regress,
)

logger = logging.getLogger(__name__)


class BaseTrace(BaseModel):
    x: pd.Series | NDArray[Any] | None = None
    y: pd.Series | NDArray[Any] | None = None
    name: str
    opacity: float | None = None
    legendgroup: str | None = None
    showlegend: bool | None = None

    @staticmethod
    def get_error_bars(
        trace_data: TraceData,
    ) -> list[dict[str, Any] | None]:
        """Computes error bars.
        `Upper` and `lower` length are calculated relative to the underlying data.
        """

        error_parameters = [
            (
                {
                    "type": "data",
                    "array": np.array([error[1] for error in error_data]) - underlying_data,
                    "arrayminus": underlying_data - np.array([error[0] for error in error_data]),
                    "visible": True,
                }
                if error_data is not None
                else None
            )
            for error_data, underlying_data in zip(
                [trace_data.error_x, trace_data.error_y, trace_data.error_z],
                [
                    trace_data.x_values,
                    trace_data.y_values,
                    trace_data.z_values,
                ],
                strict=True,
            )
        ]

        return error_parameters


class _ScatterBaseTrace(BaseTrace):
    marker: dict[str, Any] | None = None
    mode: TraceMode | None = None
    error_x: dict[str, Any] | None = None
    error_y: dict[str, Any] | None = None
    text: str | pd.Series | None = None
    textposition: str | None = None
    hoverinfo: str = "x+y+name+text"

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        mode: TraceMode | None,
    ) -> "_ScatterBaseTrace":
        error_x_data, error_y_data, _ = cls.get_error_bars(trace_data)

        return cls(
            x=trace_data.x_values,
            y=trace_data.y_values,
            name=trace_name,
            opacity=color_specifier.opacity,
            text=trace_data.text_data,
            mode=mode,
            error_x=error_x_data,
            error_y=error_y_data,
            marker={
                "size": trace_data.size_data,
                "color": (
                    color_specifier.format_color_data(trace_data.color_data)
                    if trace_data.color_data is not None
                    else trace_color
                ),
                "opacity": trace_data.opacity_data,
                "symbol": trace_data.marker_data,
                "coloraxis": color_specifier.coloraxis_reference,
            },
            legendgroup=trace_name,
        )


class _DensityTrace(BaseTrace, metaclass=ABCMeta):
    z: pd.Series | NDArray[Any]
    coloraxis: str | None = None
    zmin: float | None = None
    zmax: float | None = None
    text: str | pd.Series | None = None


class HeatmapTrace(_DensityTrace):
    hoverinfo: str = "x+y+z+text"
    colorbar: dict[str, Any] | None = None
    colorscale: str | list[list[str | float]] | None = None
    text: str | pd.Series | None = None

    @classmethod
    def build_histmap_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        color_specifier: ColorSpecifier,
        jointplot_specifier: JointplotSpecifier,
    ) -> "HeatmapTrace":
        anchor_values, hist, bin_centers = jointplot_specifier.compute_histmap(trace_data)

        return cls(
            x=(
                anchor_values
                if jointplot_specifier.plot_type is JointplotType.X_HISTMAP
                else bin_centers
            ),
            y=(
                bin_centers
                if jointplot_specifier.plot_type is JointplotType.X_HISTMAP
                else anchor_values
            ),
            z=hist,
            name=f"{trace_name} {anchor_values.name} histmap",
            opacity=color_specifier.opacity,
            text=trace_data.text_data,
            zmin=color_specifier.zmin,
            zmax=color_specifier.zmax,
            colorscale=color_specifier.build_colorscale(hist),
            colorbar=color_specifier.build_colorbar(hist),
            legendgroup=trace_name,
        )

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        color_specifier: ColorSpecifier,
    ) -> "HeatmapTrace":
        return cls(
            x=trace_data.x_values,
            y=trace_data.y_values,
            z=trace_data.z_values,
            zmin=color_specifier.zmin,
            zmax=color_specifier.zmax,
            coloraxis=color_specifier.coloraxis_reference,
            name=trace_name,
            opacity=color_specifier.opacity,
            text=trace_data.text_data,
            legendgroup=trace_name,
        )


class ScatterTrace(_ScatterBaseTrace):
    hoverinfo: str = "x+y+name+text"
    line: dict[str, Any] | None = None
    fill: str | None = None
    fillcolor: str | None = None

    @classmethod
    def build_id_line(cls, x_values: pd.Series, y_values: pd.Series) -> "ScatterTrace":
        line_data = pd.Series(
            (
                min(x_values.min(), y_values.min()),
                max(x_values.max(), y_values.max()),
            )
        )
        return cls(
            x=line_data,
            y=line_data,
            name="45° id line",
            mode=TraceMode.LINES,
            line={
                "color": constants.DEFAULT_ID_LINE_COLOR,
                "width": constants.DEFAULT_ID_LINE_WIDTH,
                "dash": constants.DEFAULT_ID_LINE_DASH,
            },
        )

    @classmethod
    def build_lower_error_trace(
        cls, trace_data: TraceData, trace_name: str, trace_color: str
    ) -> "ScatterTrace":
        if trace_data.shaded_error is None:
            raise ValueError("`trace_data.shaded_error` can not be `None`")

        return cls(
            x=trace_data.x_values,
            y=trace_data.shaded_error.apply(lambda x: x[0]),
            name=f"{trace_name} {trace_data.shaded_error.name} lower bound",
            mode=TraceMode.LINES,
            line={"width": 0},
            fill="tonexty",
            fillcolor=set_rgb_alpha(trace_color, constants.SHADED_ERROR_ALPHA),
            legendgroup=trace_name,
            showlegend=False,
        )

    @classmethod
    def build_upper_error_trace(
        cls, trace_data: TraceData, trace_name: str, trace_color: str
    ) -> "ScatterTrace":
        if trace_data.shaded_error is None:
            raise ValueError("`trace_data.shaded_error` can not be `None`")
        return cls(
            x=trace_data.x_values,
            y=trace_data.shaded_error.apply(lambda x: x[1]),
            name=f"{trace_name} {trace_data.shaded_error.name} upper bound",
            mode=TraceMode.LINES,
            marker={"size": trace_data.size_data, "color": trace_color},
            line={"width": 0},
            fillcolor=set_rgb_alpha(trace_color, constants.SHADED_ERROR_ALPHA),
            legendgroup=trace_name,
            showlegend=False,
        )

    @classmethod
    def build_regression_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        regression_type: RegressionType,
    ) -> "ScatterTrace":
        if trace_data.x_values is None or trace_data.y_values is None:
            raise ValueError("`trace_data.x_values` and `trace_data.x_values` can not be `None`")

        if regression_type is RegressionType.LINEAR:
            p, r2, (x_grid, y_fit) = regress(trace_data.x_values, trace_data.y_values, affine_func)
            regression_legend = f"alpha={p[0]:.2f}, r={np.sqrt(r2):.2f}"
        elif regression_type is RegressionType.EXPONENTIAL:
            p, r2, (x_grid, y_fit) = exponential_regress(trace_data.x_values, trace_data.y_values)
            regression_legend = f"R2={r2:.2f}"
        elif regression_type is RegressionType.INVERSE:
            p, r2, (x_grid, y_fit) = regress(trace_data.x_values, trace_data.y_values, inverse_func)
            regression_legend = f"R2={r2:.2f}"

        return cls(
            x=pd.Series(x_grid),
            y=pd.Series(y_fit),
            name=f"{trace_name} {regression_type.value} fit: {regression_legend}",
            mode=TraceMode.LINES,
            marker={"color": trace_color},
            line={"dash": constants.DEFAULT_REGRESSION_LINE_DASH},
            textposition="bottom center",
            legendgroup=trace_name,
            opacity=constants.DEFAULT_REGRESSION_LINE_OPACITY,
        )

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        mode: TraceMode | None,
    ) -> "ScatterTrace":
        return cls(
            **super()
            .build_trace(
                trace_data=trace_data,
                trace_name=trace_name,
                trace_color=trace_color,
                color_specifier=color_specifier,
                mode=mode,
            )
            .model_dump()
        )


class Scatter3DTrace(_ScatterBaseTrace):
    hoverinfo: str = "x+y+z+name+text"
    z: pd.Series | NDArray[Any]
    error_z: dict[str, Any] | None = None

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        mode: TraceMode | None,
    ) -> "Scatter3DTrace":
        scatter_trace = _ScatterBaseTrace.build_trace(
            trace_data=trace_data,
            trace_name=trace_name,
            trace_color=trace_color,
            color_specifier=color_specifier,
            mode=mode,
        )

        # error data
        _, _, error_z_data = cls.get_error_bars(trace_data)

        scatter3d_trace = deep_update(
            scatter_trace.model_dump(),
            {
                "z": trace_data.z_values,
                "error_z": error_z_data,
                "marker": {
                    "line": {
                        "color": constants.DEFAULT_MARKER_LINE_COLOR,
                        "width": constants.DEFAULT_MARKER_LINE_WIDTH,
                    }
                },
            },
        )

        return cls.model_validate(scatter3d_trace)


class _CategoricalTrace(BaseTrace, metaclass=ABCMeta):
    hoverinfo: str = "x+y+name+text"
    marker: dict[str, Any] | None = None
    text: str | pd.Series | None = None

    @classmethod
    @abstractmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
    ) -> "_CategoricalTrace":
        return cls(
            x=trace_data.x_values,
            y=trace_data.y_values,
            name=trace_name,
            text=trace_data.text_data,
            opacity=color_specifier.opacity,
            marker={
                "size": trace_data.size_data,
                "color": (
                    color_specifier.format_color_data(trace_data.color_data)
                    if trace_data.color_data is not None
                    else trace_color
                ),
                "opacity": trace_data.opacity_data,
                "symbol": trace_data.marker_data,
            },
        )


class StripTrace(_CategoricalTrace):
    mode: str = TraceMode.MARKERS

    @staticmethod
    def get_x_strip_map(x_values: pd.Series) -> dict[str, Any]:
        def _cast_strip_coordinates(x_coord: Any) -> Any:
            if np.issubdtype(type(x_coord), np.datetime64):
                return pd.Timestamp(x_coord)
            return x_coord

        x_dict: dict[str, Any] = {}
        for i, x_level in enumerate(np.sort(x_values.dropna().unique()), 1):
            x_dict[_cast_strip_coordinates(x_level)] = i

        return x_dict

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
    ) -> "StripTrace":
        return cls(
            **super().build_trace(trace_data, trace_name, trace_color, color_specifier).model_dump()
        )


class BoxTrace(_CategoricalTrace):
    boxmean: bool = True

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
    ) -> "BoxTrace":
        return cls(
            **super().build_trace(trace_data, trace_name, trace_color, color_specifier).model_dump()
        )


class ViolinTrace(_CategoricalTrace):
    meanline_visible: bool = True
    scalemode: str = "width"

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
    ) -> "ViolinTrace":
        return cls(
            **super().build_trace(trace_data, trace_name, trace_color, color_specifier).model_dump()
        )


class BarTrace(BaseTrace):
    hoverinfo: str = "x+y+name+text"
    marker: dict[str, Any] | None = None
    error_y: dict[str, Any] | None = None
    text: str | pd.Series | None = None
    textposition: str | None = None

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData | AggregationTraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
    ) -> "BarTrace":
        _, error_y_data, _ = cls.get_error_bars(trace_data)

        return cls(
            x=trace_data.x_values,
            y=trace_data.y_values,
            name=trace_name,
            opacity=color_specifier.opacity,
            text=trace_data.text_data,
            error_y=error_y_data,
            marker={
                "color": (
                    color_specifier.format_color_data(trace_data.color_data)
                    if trace_data.color_data is not None
                    else trace_color
                ),
                "opacity": trace_data.opacity_data,
                "coloraxis": color_specifier.coloraxis_reference,
            },
            legendgroup=trace_name,
        )


class StepHistogramTrace(BaseTrace):
    line: dict[str, Any]
    hoverinfo: str = "all"
    mode: TraceMode = TraceMode.LINES

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        histogram_specifier: HistogramSpecifier,
    ) -> "StepHistogramTrace":
        histogram_data = getattr(trace_data, TRACE_DIMENSION_MAP[histogram_specifier.dimension])
        hist, bin_edges, binsize = histogram_specifier.histogram(histogram_data)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        return cls(
            x=bin_centers if histogram_specifier.dimension is DataDimension.X else hist,
            y=hist if histogram_specifier.dimension is DataDimension.X else bin_centers,
            name=f"{trace_name} {histogram_data.name}",
            line={
                "color": (
                    trace_data.color_data if trace_data.color_data is not None else trace_color
                ),
                "shape": "hvh" if histogram_specifier.dimension is DataDimension.X else "vhv",
            },
            opacity=color_specifier.opacity,
            legendgroup=trace_name,
        )


class RugTrace(BaseTrace):
    hoverinfo: str
    line: dict[str, Any] | None = None
    mode: TraceMode = TraceMode.LINES
    showlegend: bool = False
    text: str | pd.Series | None = None

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        histogram_specifier: HistogramSpecifier,
    ) -> "RugTrace":
        rug_data = getattr(trace_data, TRACE_DIMENSION_MAP[histogram_specifier.dimension])

        rug_coord = np.tile(rug_data, (2, 1)).transpose()
        rug_coord_grid = np.concatenate(
            (rug_coord, np.tile(None, (len(rug_coord), 1))),  # type:ignore
            axis=1,
        ).ravel()

        hist, _, _ = histogram_specifier.histogram(data=rug_data)

        rug_length_coord = np.tile(np.array([0, 0.1 * max(hist)]), (len(rug_coord), 1))
        rug_length_grid = np.concatenate(
            (
                rug_length_coord,
                np.tile(None, (len(rug_length_coord), 1)),  # type:ignore
            ),
            axis=1,
        ).ravel()

        return cls(
            x=(
                rug_coord_grid
                if histogram_specifier.dimension is DataDimension.X
                else rug_length_grid
            ),
            y=(
                rug_length_grid
                if histogram_specifier.dimension is DataDimension.X
                else rug_coord_grid
            ),
            name=f"{trace_name} {rug_data.name} raw observations",
            hoverinfo="x+text" if histogram_specifier.dimension is DataDimension.X else "y+text",
            line={
                "color": (
                    trace_data.color_data if trace_data.color_data is not None else trace_color
                ),
                "width": 1,
            },
            legendgroup=trace_name,
        )


class HistogramTrace(BaseTrace):
    marker: dict[str, Any] | None = None
    cumulative: dict[str, Any] | None = None
    xbins: dict[str, Any] | None = None
    histnorm: HistogramNormType | None = None
    hoverinfo: str = "all"

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        histogram_specifier: HistogramSpecifier,
    ) -> "HistogramTrace":
        histogram_data = getattr(trace_data, TRACE_DIMENSION_MAP[histogram_specifier.dimension])
        bin_edges, bin_size = histogram_specifier.histogram_bin_edges(histogram_data)

        return cls(
            x=histogram_data if histogram_specifier.dimension is DataDimension.X else None,
            y=histogram_data if histogram_specifier.dimension is DataDimension.Y else None,
            name=f"{trace_name} distribution",
            opacity=color_specifier.opacity,
            legendgroup=trace_name,
            marker={
                "color": trace_data.color_data if trace_data.color_data is not None else trace_color
            },
            cumulative={"enabled": histogram_specifier.cumulative},
            xbins={"start": bin_edges[0], "end": bin_edges[-1], "size": bin_size},
            histnorm=histogram_specifier.histnorm,
        )


class Histogram2dTrace(BaseTrace):
    marker: dict[str, Any] | None = None
    xbins: dict[str, Any] | None = None
    ybins: dict[str, Any] | None = None
    colorbar: dict[str, Any] | None = None
    colorscale: str | list[list[str | float]] | None = None
    histnorm: HistogramNormType | None = None
    hoverinfo: str = "all"

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        jointplot_specifier: JointplotSpecifier,
    ) -> "Histogram2dTrace":
        (
            hist,
            (xbin_edges, ybin_edges),
            (xbin_size, ybin_size),
        ) = jointplot_specifier.histogram2d(
            data=pd.concat([trace_data.x_values, trace_data.y_values], axis=1)
        )

        return cls(
            x=trace_data.x_values,
            y=trace_data.y_values,
            name=f"{trace_name} density",
            opacity=color_specifier.opacity,
            legendgroup=trace_name,
            xbins={"start": xbin_edges[0], "end": xbin_edges[-1], "size": xbin_size},
            ybins={"start": ybin_edges[0], "end": ybin_edges[-1], "size": ybin_size},
            coloraxis=color_specifier.coloraxis_reference,
            colorscale=(
                color_specifier.build_colorscale(hist)
                if color_specifier.coloraxis_reference is None
                else None
            ),
            colorbar=(
                color_specifier.build_colorbar(hist)
                if color_specifier.coloraxis_reference is None
                else None
            ),
            histnorm=jointplot_specifier.histogram_specifier[  # type: ignore
                DataDimension.X
            ].histnorm,
        )


class KdeTrace(BaseTrace):
    hoverinfo: str = "none"
    line: dict[str, Any]
    mode: TraceMode = TraceMode.LINES
    showlegend: bool = False

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        histogram_specifier: HistogramSpecifier,
    ) -> "KdeTrace":
        histogram_data = getattr(trace_data, TRACE_DIMENSION_MAP[histogram_specifier.dimension])
        bin_edges, bin_size = histogram_specifier.histogram_bin_edges(histogram_data)

        grid = np.linspace(
            np.floor(bin_edges.min()),
            np.ceil(bin_edges.max()),
            constants.N_GRID_POINTS,
        )
        kde = kde_1d(histogram_data, grid)
        color = trace_data.color_data if trace_data.color_data is not None else trace_color

        return cls(
            x=grid if histogram_specifier.dimension is DataDimension.X else kde,
            y=kde if histogram_specifier.dimension is DataDimension.X else grid,
            name=f"{trace_name} pdf",
            line={"color": set_rgb_alpha(color, color_specifier.opacity or 1)},
            legendgroup=trace_name,
        )


class HistogramLineTrace(BaseTrace):
    hoverinfo: str
    line: dict[str, Any]
    mode: TraceMode = TraceMode.LINES
    showlegend: bool = True

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        histogram_specifier: HistogramSpecifier,
        hline_coordinates: tuple[str, float] | None = None,
        vline_coordinates: tuple[str, float] | None = None,
    ) -> "HistogramLineTrace":
        if vline_coordinates is not None:
            vline_name, vline_data = vline_coordinates
            x_data = np.repeat(vline_data, 2)
            if trace_data.x_values is not None:
                hist, _, _ = histogram_specifier.histogram(trace_data.x_values)
                y_data = np.array([0, max(hist)])
            else:
                if trace_data.y_values is None:
                    raise ValueError("`trace_data.y_values` can not be `None`")
                y_data = np.sort(trace_data.y_values)[[0, -1]]
            name = f"{trace_name} {vline_name}={vline_data:.2f}"
            hoverinfo = "x+name"

        elif hline_coordinates is not None:
            hline_name, hline_data = hline_coordinates
            y_data = np.repeat(hline_data, 2)
            if trace_data.y_values is not None:
                hist, _, _ = histogram_specifier.histogram(trace_data.y_values)
                x_data = np.array([0, max(hist)])
            else:
                if trace_data.x_values is None:
                    raise ValueError("`trace_data.x_values` can not be `None`")
                x_data = np.sort(trace_data.x_values)[[0, -1]]
            name = f"{trace_name} {hline_name}={hline_data:.2f}"
            hoverinfo = "y+name"
        else:
            raise Exception(f"Missing line coordinates for {HistogramLineTrace.__name__} object")

        return cls(
            x=x_data,
            y=y_data,
            name=name,
            line={
                "color": (
                    trace_data.color_data if trace_data.color_data is not None else trace_color
                ),
                "dash": "dot",
            },
            hoverinfo=hoverinfo,
            legendgroup=trace_name,
        )


class ContourTrace(_DensityTrace):
    colorscale: str | list[list[str | float]] | None = None
    hoverinfo: str = "all"
    ncontours: int
    reversescale: bool = True
    showscale: bool = False

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        color_specifier: ColorSpecifier,
        jointplot_specifier: JointplotSpecifier,
    ) -> "ContourTrace":
        if trace_data.x_values is None or trace_data.y_values is None:
            raise ValueError("`trace_data.x_values` and `trace_data.x_values` can not be `None`")
        # X grid
        bin_edges, binsize = jointplot_specifier.histogram_specifier[  # type: ignore
            DataDimension.X
        ].histogram_bin_edges(trace_data.x_values)
        x_grid = np.linspace(
            np.floor(bin_edges.min()),
            np.ceil(bin_edges.max()),
            constants.N_GRID_POINTS,
        )

        # Y grid
        bin_edges, binsize = jointplot_specifier.histogram_specifier[  # type: ignore
            DataDimension.Y
        ].histogram_bin_edges(trace_data.y_values)
        y_grid = np.linspace(
            np.floor(bin_edges.min()),
            np.ceil(bin_edges.max()),
            constants.N_GRID_POINTS,
        )

        z_data = kde_2d(trace_data.x_values, trace_data.y_values, x_grid, y_grid)

        return cls(
            x=x_grid,
            y=y_grid,
            z=z_data,
            zmin=color_specifier.zmin,
            zmax=color_specifier.zmax,
            coloraxis=(
                color_specifier.coloraxis_reference
                if color_specifier.coloraxis_reference is not None
                else None
            ),
            colorscale=color_specifier.build_colorscale(z_data),
            name=f"{trace_name} {trace_data.y_values.name} vs {trace_data.x_values.name} KDE",
            ncontours=constants.N_CONTOUR_LINES,
            legendgroup=trace_name,
        )
