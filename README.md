[StatsPlot](https://github.com/Parici75/statsplot) is a Python data visualization library based on Plotly. It provides a high-level interface for drawing attractive and interactive statistical data vizualization plots.

The development of this library paralleled the fantastic [plotly.express](https://plotly.com/python/plotly-express/) API.

Philosophy
-
Compared to `plotly.express` API, StatsPlots :
- respects common conventions of statistical visualization (e.g., histograms are not barplots).
- processes color coding scheme, trace slicer and plot dimensions independently. 

This flexibility allows to leverage the powerful interactivity offered by plotly.js without compromising statistical intelligibility for aesthetic choices, or vice-versa.

Examples
-
Main features of the API are demonstrated in a demo [notebook](https://nbviewer.org/github/Parici75/statsplot/blob/main/docs/notebooks/statsplot_demo.ipynb). 

Installation
-
Clone and install with pip :

`pip install .`

Documentation
-
Details of the public API can be be found in the [documentation](https://parici75.github.io/statsplot).


Requirements
-
- [Plotly](https://plotly.com/python/)
- [Pymodules](https://github.com/Parici75/pymodules), a collection of general purpose utility functions.


Author
-
[Benjamin Roland](benjamin.roland@hotmail.fr)