import plotly

# Template
DEFAULT_TEMPLATE = "statsplotly_template"

# Colorscale
N_COLORSCALE_COLORS = 256
DEFAULT_COLOR_PALETTE = "viridis"
SEABORN_DEFAULT_CONTINUOUS_COLOR_PALETTE = "rocket"
DEFAULT_KDE_COLOR_PALETTE = "magma"
SHADED_ERROR_ALPHA = 0.2
DEFAULT_HISTOGRAM_OPACITY = 0.8
DEFAULT_OVERLAID_HISTOGRAM_OPACITY = 0.5
BUILTIN_COLORSCALES = plotly.colors.named_colorscales()
CSS_COLORS = [
    "aliceblue",
    "antiquewhite",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgrey",
    "darkgreen",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dimgrey",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "gold",
    "goldenrod",
    "gray",
    "grey",
    "green",
    "greenyellow",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgrey",
    "lightgreen",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightslategrey",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "red",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "slategrey",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "whitesmoke",
    "yellow",
    "yellowgreen",
]

# Markers
DEFAULT_MARKER_SIZE = 6
MIN_MARKER_SIZE = 4
MAX_MARKER_SIZE = 24
DEFAULT_MARKER_LINE_COLOR = "grey"
DEFAULT_MARKER_LINE_WIDTH = 2
DEFAULT_TRANSPARENCE_OPACITY = 0.8
DEFAULT_REGRESSION_LINE_OPACITY = 1
DEFAULT_REGRESSION_LINE_DASH = "dash"
DEFAULT_HOVERMODE = "closest"
DEFAULT_STRIPPLOT_JITTER = 1

DEFAULT_ID_LINE_DASH = "longdash"
DEFAULT_ID_LINE_COLOR = "grey"
DEFAULT_ID_LINE_WIDTH = 2

# Statistics
N_GRID_POINTS = 100
N_CONTOUR_LINES = 20
CI_ALPHA = 0.05
IQR = [0.25, 0.75]
DEFAULT_HISTOGRAM_BIN_COMPUTATION_METHOD = "scott"
DEFAULT_KDE_BANDWIDTH = 0.2

# Axis parameters
FIGURE_TITLEFONT = {"family": "Arial", "size": 20}
AXIS_TITLEFONT = {"family": "Arial", "size": 20}
TICKFONT = {"family": "Arial", "size": 18}
SCENE_TICKFONT = {"family": "Arial", "size": 16}
AXES_HEIGHT = 600
AXES_WIDTH = 600
MIN_CAPITALIZE_LENGTH = 3
RANGE_PADDING_FACTOR = 0.1

# Layout parameters
LAYOUT_UPDATE_MENUS_TYPE = "dropdown"
LAYOUT_UPDATE_MENUS_DIRECTION = "down"

# Subplot parameters
COLORBAR_THICKNESS_SCALING_FACTOR = 30
COLORBAR_XOFFSET = 1.2
COLORBAR_REDUCTION_FACTOR = 0.5
COLORBAR_BOTTOM_POSITION = -0.5
