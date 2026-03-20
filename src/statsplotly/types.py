from typing import Literal

TraceModeLiteral = Literal["markers", "lines", "markers+lines", "lines+text"]
AxisFormatLiteral = Literal["square", "fixed_ratio", "equal", "id_line"]
NormalizationTypeLiteral = Literal["center", "minmax", "zscore"]
RegressionTypeLiteral = Literal["linear", "exponential", "inverse"]
ErrorBarLiteral = Literal["sem", "iqr", "std", "geo_std", "bootstrap"]
AggregationTypeLiteral = Literal[
    "mean", "geo_mean", "count", "median", "percent", "fraction", "sum"
]
CategoricalPlotTypeLiteral = Literal["box", "violin", "strip"]
PlotOrientationTypeLiteral = Literal["horizontal", "vertical"]
BarModeLiteral = Literal["stack", "group", "overlay", "relative"]

CentralTendencyTypeLiteral = Literal["mean", "median", "mode"]
HistogramNormTypeLiteral = Literal["", "percent", "probability", "probability density"]
JointplotTypeLiteral = Literal[
    "scatter", "kde", "scatter+kde", "x_histmap", "y_histmap", "histogram"
]

MarginalPlotDimensionLiteral = Literal["x", "y", "all"]
SharedGridAxisLiteral = Literal["cols", "rows", "all"]
