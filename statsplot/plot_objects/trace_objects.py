from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pydantic.utils import deep_update

from pymodules.pandas_utils import unique_non_null
from statsplot import constants
from statsplot.plot_specifiers.color import ColorSpecifier
from statsplot.plot_specifiers.data import (
    BaseModel,
    TraceData,
    DataDimension,
    RegressionType,
    TRACE_DIMENSION_MAP,
    HistogramNormType,
)
from statsplot.plot_specifiers.trace import (
    TraceMode,
    JointplotType,
    HistogramSpecifier,
    JointplotSpecifier,
)
from statsplot.utils.colors_utils import set_rgb_alpha
from statsplot.utils.stats_utils import (
    regress,
    exponential_regress,
    affine_func,
    inverse_func,
    kde_1d,
    kde_2d,
)


class BaseTrace(BaseModel):
    x: pd.Series | np.ndarray | None
    y: pd.Series | np.ndarray | None
    name: str
    opacity: float | None
    legendgroup: str | None
    showlegend: bool | None

    @staticmethod
    def get_error_bars(
        trace_data: TraceData,
    ) -> List[Dict[str, Any] | None]:
        """Computes error bars.
        `Upper` and `lower` length are calculated relative to the underlying data.
        """

        error_parameters = [
            {
                "type": "data",
                "array": np.array([error[1] for error in error_data])
                - underlying_data,
                "arrayminus": underlying_data
                - np.array([error[0] for error in error_data]),
                "visible": True,
            }
            if error_data is not None
            else None
            for error_data, underlying_data in zip(
                [trace_data.error_x, trace_data.error_y, trace_data.error_z],
                [
                    trace_data.x_values,
                    trace_data.y_values,
                    trace_data.z_values,
                ],
            )
        ]

        return error_parameters


class _ScatterBaseTrace(BaseTrace):
    marker: Dict[str, Any] | None
    mode: TraceMode | None
    error_x: Dict[str, Any] | None
    error_y: Dict[str, Any] | None
    text: str | pd.Series | None
    textposition: str | None
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
                "color": trace_data.color_data
                if trace_data.color_data is not None
                else trace_color,
                "symbol": trace_data.marker_data,
                "coloraxis": color_specifier.coloraxis_reference,
            },
            legendgroup=trace_name,
        )


class _DensityTrace(BaseTrace, metaclass=ABCMeta):
    z: pd.Series | np.ndarray
    coloraxis: str | None
    zmin: float | None
    zmax: float | None
    text: str | pd.Series | None


class HeatmapTrace(_DensityTrace):
    hoverinfo: str = "x+y+z+text"
    colorbar: Dict[str, Any] | None
    colorscale: str | List | None
    text: str | pd.Series | None

    @classmethod
    def build_histmap_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        color_specifier: ColorSpecifier,
        jointplot_specifier: JointplotSpecifier,
    ) -> "HeatmapTrace":
        anchor_values, hist, bin_centers = jointplot_specifier.compute_histmap(
            trace_data
        )

        return cls(
            x=anchor_values
            if jointplot_specifier.plot_type is JointplotType.X_HISTMAP
            else bin_centers,
            y=bin_centers
            if jointplot_specifier.plot_type is JointplotType.X_HISTMAP
            else anchor_values,
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
    line: Dict[str, Any] | None
    fill: str | None
    fillcolor: str | None

    @classmethod
    def build_id_line(
        cls, x_values: pd.Series, y_values: pd.Series
    ) -> "ScatterTrace":
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
            line=dict(
                color=constants.DEFAULT_ID_LINE_COLOR,
                width=constants.DEFAULT_ID_LINE_WIDTH,
                dash=constants.DEFAULT_ID_LINE_DASH,
            ),
        )

    @classmethod
    def build_lower_error_trace(
        cls, trace_data: TraceData, trace_name: str, trace_color: str
    ) -> "ScatterTrace":
        assert trace_data.shaded_error is not None
        return cls(
            x=trace_data.x_values,
            y=trace_data.shaded_error.apply(lambda x: x[0]),
            name=f"{trace_name} {trace_data.shaded_error.name} lower bound",
            mode=TraceMode.LINES,
            line=dict(width=0),
            fill="tonexty",
            fillcolor=set_rgb_alpha(trace_color, constants.SHADED_ERROR_ALPHA),
            legendgroup=trace_name,
            showlegend=False,
        )

    @classmethod
    def build_upper_error_trace(
        cls, trace_data: TraceData, trace_name: str, trace_color: str
    ) -> "ScatterTrace":
        assert trace_data.shaded_error is not None
        return cls(
            x=trace_data.x_values,
            y=trace_data.shaded_error.apply(lambda x: x[1]),
            name=f"{trace_name} {trace_data.shaded_error.name} upper bound",
            mode=TraceMode.LINES,
            marker={"size": trace_data.size_data, "color": trace_color},
            line=dict(width=0),
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
        assert trace_data.x_values is not None
        assert trace_data.y_values is not None

        if regression_type is RegressionType.LINEAR:
            p, r2, (x_grid, y_fit) = regress(
                trace_data.x_values, trace_data.y_values, affine_func
            )
            regression_legend = f"alpha={p[0]:.2f}, r={np.sqrt(r2):.2f}"
        elif regression_type is RegressionType.EXPONENTIAL:
            p, r2, (x_grid, y_fit) = exponential_regress(
                trace_data.x_values, trace_data.y_values
            )
            regression_legend = f"R2={r2:.2f}"
        elif regression_type is RegressionType.INVERSE:
            p, r2, (x_grid, y_fit) = regress(
                trace_data.x_values, trace_data.y_values, inverse_func
            )
            regression_legend = f"R2={r2:.2f}"

        return cls(
            x=pd.Series(x_grid),
            y=pd.Series(y_fit),
            name=f"{trace_name} {regression_type} fit: {regression_legend}",
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
            .dict()
        )


class Scatter3DTrace(_ScatterBaseTrace):
    hoverinfo: str = "x+y+z+name+text"
    z: pd.Series | np.ndarray
    error_z: Dict[str, Any] | None

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
            scatter_trace.dict(),
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

        return cls.parse_obj(scatter3d_trace)


class _CategoricalTrace(BaseTrace, metaclass=ABCMeta):
    hoverinfo: str = "x+y+name+text"
    marker: Dict[str, Any] | None
    text: str | pd.Series | None

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
                "color": trace_data.color_data
                if trace_data.color_data is not None
                else trace_color,
            },
        )


class StripTrace(_CategoricalTrace):
    mode: str = TraceMode.MARKERS

    @staticmethod
    def get_x_strip_map(x_values: pd.Series) -> Dict[str, Any]:
        x_dict: Dict[str, Any] = {}
        for i, x_level in enumerate(np.sort(unique_non_null(x_values)), 1):
            if np.issubdtype(type(x_level), np.datetime64):
                x_level = pd.Timestamp(x_level)
            x_dict[x_level] = i

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
            **super()
            .build_trace(trace_data, trace_name, trace_color, color_specifier)
            .dict()
        )


class BoxTrace(_CategoricalTrace):
    boxmean = True

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
    ) -> "BoxTrace":
        return cls(
            **super()
            .build_trace(trace_data, trace_name, trace_color, color_specifier)
            .dict()
        )


class ViolinTrace(_CategoricalTrace):
    meanline_visible = True
    scalemode = "width"

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
    ) -> "ViolinTrace":
        return cls(
            **super()
            .build_trace(trace_data, trace_name, trace_color, color_specifier)
            .dict()
        )


class BarTrace(BaseTrace):
    hoverinfo: str = "x+y+name+text"
    marker: Dict[str, Any] | None
    error_y: Dict[str, Any] | None
    text: str | pd.Series | None
    textposition: str | None

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
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
                "color": trace_data.color_data
                if trace_data.color_data is not None
                else trace_color,
                "coloraxis": color_specifier.coloraxis_reference,
            },
            legendgroup=trace_name,
        )


class StepHistogramTrace(BaseTrace):
    line: Dict[str, Any]
    hoverinfo = "all"

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        histogram_specifier: HistogramSpecifier,
    ) -> "StepHistogramTrace":
        assert histogram_specifier.dimension is not None
        histogram_data = getattr(
            trace_data, TRACE_DIMENSION_MAP[histogram_specifier.dimension]
        )
        hist, bin_edges, binsize = histogram_specifier.histogram(
            histogram_data
        )
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        return cls(
            x=bin_centers
            if histogram_specifier.dimension is DataDimension.X
            else hist,
            y=hist
            if histogram_specifier.dimension is DataDimension.X
            else bin_centers,
            name=f"{trace_name} {histogram_data.name}",
            mode=TraceMode.LINES,
            line=dict(
                color=trace_data.color_data
                if trace_data.color_data is not None
                else trace_color,
                shape="hvh"
                if histogram_specifier.dimension is DataDimension.X
                else "vhv",
            ),
            opacity=color_specifier.opacity,
            text=trace_data.text_data,
            legendgroup=trace_name,
        )


class RugTrace(BaseTrace):
    line: Dict[str, Any] | None
    showlegend = False
    text: str | pd.Series | None

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        histogram_specifier: HistogramSpecifier,
    ) -> "RugTrace":
        assert histogram_specifier.dimension is not None
        rug_data = getattr(
            trace_data, TRACE_DIMENSION_MAP[histogram_specifier.dimension]
        )

        rug_coord = np.tile(rug_data, (2, 1)).transpose()
        rug_coord_grid = np.concatenate(
            (rug_coord, np.tile(None, (len(rug_coord), 1))),  # type:ignore
            axis=1,
        ).ravel()

        hist, _, _ = histogram_specifier.histogram(data=rug_data)

        rug_length_coord = np.tile(
            np.array([0, 0.1 * max(hist)]), (len(rug_coord), 1)
        )
        rug_length_grid = np.concatenate(
            (
                rug_length_coord,
                np.tile(None, (len(rug_length_coord), 1)),  # type:ignore
            ),
            axis=1,
        ).ravel()

        return cls(
            x=rug_coord_grid
            if histogram_specifier.dimension is DataDimension.X
            else rug_length_grid,
            y=rug_length_grid
            if histogram_specifier.dimension is DataDimension.X
            else rug_coord_grid,
            name=f"{trace_name} {rug_data.name} raw observations",
            mode=TraceMode.LINES,
            hoverinfo="x+text"
            if histogram_specifier.dimension is DataDimension.X
            else "y+text",
            line=dict(
                color=trace_data.color_data
                if trace_data.color_data is not None
                else trace_color,
                width=1,
            ),
            legendgroup=trace_name,
        )


class HistogramTrace(BaseTrace):
    marker: Dict[str, Any] | None
    cumulative: Dict[str, Any] | None
    xbins: Dict[str, Any] | None
    histnorm: HistogramNormType | None
    hoverinfo = "all"

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        histogram_specifier: HistogramSpecifier,
    ) -> "HistogramTrace":
        histogram_data = getattr(
            trace_data, TRACE_DIMENSION_MAP[histogram_specifier.dimension]
        )
        bin_edges, bin_size = histogram_specifier.histogram_bin_edges(
            histogram_data
        )

        return cls(
            x=histogram_data
            if histogram_specifier.dimension is DataDimension.X
            else None,
            y=histogram_data
            if histogram_specifier.dimension is DataDimension.Y
            else None,
            name=f"{trace_name} {histogram_data.name}",
            opacity=color_specifier.opacity,
            legendgroup=trace_name,
            marker={
                "color": trace_data.color_data
                if trace_data.color_data is not None
                else trace_color
            },
            cumulative=dict(enabled=histogram_specifier.cumulative),
            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=bin_size),
            histnorm=histogram_specifier.histnorm,
        )


class Histogram2dTrace(BaseTrace):
    marker: Dict[str, Any] | None
    xbins: Dict[str, Any] | None
    ybins: Dict[str, Any] | None
    colorbar: Dict[str, Any] | None
    colorscale: str | List | None
    histnorm: HistogramNormType | None
    hoverinfo = "all"

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
            name=f"{trace_name}",
            opacity=color_specifier.opacity,
            legendgroup=trace_name,
            xbins=dict(
                start=xbin_edges[0], end=xbin_edges[-1], size=xbin_size
            ),
            ybins=dict(
                start=ybin_edges[0], end=ybin_edges[-1], size=ybin_size
            ),
            coloraxis=color_specifier.coloraxis_reference,
            colorscale=color_specifier.build_colorscale(hist)
            if color_specifier.coloraxis_reference is None
            else None,
            colorbar=color_specifier.build_colorbar(hist)
            if color_specifier.coloraxis_reference is None
            else None,
            histnorm=jointplot_specifier.histogram_specifier[ #type: ignore
                DataDimension.X
            ].histnorm,
        )


class KdeTrace(BaseTrace):
    hoverinfo = "none"
    line: Dict[str, Any]
    showlegend = False

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        histogram_specifier: HistogramSpecifier,
    ) -> "KdeTrace":
        histogram_data = getattr(
            trace_data, TRACE_DIMENSION_MAP[histogram_specifier.dimension]
        )
        bin_edges, bin_size = histogram_specifier.histogram_bin_edges(
            histogram_data
        )

        grid = np.linspace(
            np.floor(bin_edges.min()),
            np.ceil(bin_edges.max()),
            constants.N_GRID_POINTS,
        )
        kde = kde_1d(histogram_data, grid)
        color = (
            trace_data.color_data
            if trace_data.color_data is not None
            else trace_color
        )

        return cls(
            x=grid
            if histogram_specifier.dimension is DataDimension.X
            else kde,
            y=kde
            if histogram_specifier.dimension is DataDimension.X
            else grid,
            name=f"{trace_name} pdf",
            mode=TraceMode.LINES,
            line=dict(color=set_rgb_alpha(color, color_specifier.opacity)),
            legendgroup=trace_name,
        )


class HistogramLineTrace(BaseTrace):
    line: Dict[str, Any]
    mode = TraceMode.LINES
    showlegend = True

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        histogram_specifier: HistogramSpecifier,
        hline_coordinates: Tuple[str, float] | None = None,
        vline_coordinates: Tuple[str, float] | None = None,
    ) -> "HistogramLineTrace":
        if vline_coordinates is not None:
            vline_name, vline_data = vline_coordinates
            x_data = np.repeat(vline_data, 2)
            if trace_data.x_values is not None:
                hist, _, _ = histogram_specifier.histogram(trace_data.x_values)
                y_data = np.array([0, max(hist)])
            else:
                assert trace_data.y_values is not None
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
                assert trace_data.x_values is not None
                x_data = np.sort(trace_data.x_values)[[0, -1]]
            name = f"{trace_name} {hline_name}={hline_data:.2f}"
            hoverinfo = "y+name"
        else:
            raise Exception(
                f"Missing line coordinates for {HistogramLineTrace.__name__} object"
            )

        return cls(
            x=x_data,
            y=y_data,
            name=name,
            line=dict(
                color=trace_data.color_data
                if trace_data.color_data is not None
                else trace_color,
                dash="dot",
            ),
            hoverinfo=hoverinfo,
            legendgroup=trace_name,
        )


class ContourTrace(_DensityTrace):
    colorscale: str | List | None
    hoverinfo: str = "all"
    ncontours: int
    reversescale = True
    showscale = False

    @classmethod
    def build_trace(
        cls,
        trace_data: TraceData,
        trace_name: str,
        trace_color: str,
        color_specifier: ColorSpecifier,
        jointplot_specifier: JointplotSpecifier,
    ) -> "ContourTrace":
        assert trace_data.x_values is not None
        assert trace_data.y_values is not None

        # X grid
        bin_edges, binsize = jointplot_specifier.histogram_specifier[ #type: ignore
            DataDimension.X
        ].histogram_bin_edges(trace_data.x_values)
        x_grid = np.linspace(
            np.floor(bin_edges.min()),
            np.ceil(bin_edges.max()),
            constants.N_GRID_POINTS,
        )

        # Y grid
        bin_edges, binsize = jointplot_specifier.histogram_specifier[ #type: ignore
            DataDimension.Y
        ].histogram_bin_edges(trace_data.y_values)
        y_grid = np.linspace(
            np.floor(bin_edges.min()),
            np.ceil(bin_edges.max()),
            constants.N_GRID_POINTS,
        )

        z_data = kde_2d(
            trace_data.x_values, trace_data.y_values, x_grid, y_grid
        )

        return cls(
            x=x_grid,
            y=y_grid,
            z=z_data,
            zmin=color_specifier.zmin,
            zmax=color_specifier.zmax,
            coloraxis=color_specifier.coloraxis_reference
            if color_specifier.coloraxis_reference is not None
            else None,
            colorscale=color_specifier.build_colorscale(z_data),
            name=f"{trace_name} {trace_data.y_values.name} vs {trace_data.x_values.name} KDE",
            ncontours=constants.N_CONTOUR_LINES,
            legendgroup=trace_name,
        )
