[StatsPlot](https://github.com/Parici75/statsplot) is a Python data visualization library based on Plotly. It provides a high-level interface for drawing attractive and interactive statistical data visualization plots.

The development of this library paralleled the fantastic [plotly.express](https://plotly.com/python/plotly-express/) API.

Philosophy
-
Compared to `plotly.express` API, StatsPlots :
- respects common conventions of statistical visualization (e.g., histograms are not barplots).
- processes color coding scheme, trace slicer and plot dimensions independently.

This flexibility allows to leverage the powerful interactivity offered by plotly.js without compromising statistical intelligibility for aesthetic choices, or vice-versa.

![statsplot-demo](statsplot-demo.gif)

Examples
-
Main features of the API are demonstrated in a demo [notebook](https://nbviewer.org/github/Parici75/statsplot/blob/main/docs/notebooks/statsplot_demo.ipynb).

Installation
-
### Using Pip

`pip install statsplot`


Development
-
### Using Poetry
First make sure you have Poetry installed on your system (see [instruction](https://python-poetry.org/docs/#installing-with-the-official-installer)).

Then, assuming you have a Unix shell with make, create and set up a new Poetry environment :

`make init`

To make the Poetry-managed kernel available for a globally installed Jupyter :
```
$ poetry run ipython kernel install --user --name=<KERNEL_NAME>
$ jupyter notebook
```
And select the created kernel in “Kernel” -> “Change kernel”.

### Dissecting Makefile
The Makefile provides several targets to assist in development and code quality :
- `init` creates a project-specific virtual environment and installs the dependencies of the .lock file.
- `ci` launches Black, Ruff, mypy and pytest on your source code.
- `pre-commit` set up pre-commit hooks (see pre-commit [documentation](https://pre-commit.com/)).
- `clean` clears bytecode, poetry/pip caches. Use with caution.


Documentation
-
Details of the public API can be found in the [documentation](https://parici75.github.io/statsplot).


Requirements
-
- [Plotly](https://plotly.com/python/)
- [Pymodules](https://github.com/Parici75/pymodules), a collection of general purpose utility functions.


Author
-
[Benjamin Roland](benjamin.roland@hotmail.fr)
