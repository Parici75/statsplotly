[Statsplotly](https://github.com/parici75/statsplotly) is a Python data visualization library based on Plotly. It provides a high-level interface for drawing attractive and interactive statistical data visualization plots.

The inception of this library predated the fantastic [plotly.express](https://plotly.com/python/plotly-express/) API.


Philosophy
-
Compared to `plotly.express` API, `statsplotly` :
- respects common conventions of statistical visualization (e.g., histograms are not barplots).
- processes color coding scheme, trace slicer and plot dimensions independently.
- can perform standard statistical processing procedure (e.g., zscore normalization) of data under the hood.
- leverages the tidy DataFrame structure to easily style plot cues to be used as visual discriminators (e.g., marker color, symbol, size, and opacity).


This flexibility takes advantage of the powerful interactivity offered by `plotly.js` without compromising statistical intelligibility for aesthetic choices, or vice-versa.

Examples
-
Main features of the API are demonstrated in a demo [notebook](https://nbviewer.org/github/parici75/statsplotly/blob/main/docs/notebooks/statsplotly_demo.ipynb).

![statsplotly-demo](statsplotly-demo.gif)


Installation
-
### Using Pip

```bash
pip install statsplotly
```

Documentation
-
Details of the public API can be found in the [documentation](https://parici75.github.io/statsplotly).


Development
-
### Using Poetry
First make sure you have Poetry installed on your system (see [instruction](https://python-poetry.org/docs/#installing-with-the-official-installer)).

Then, assuming you have a Unix shell with make, create and set up a new Poetry environment :

```bash
make init
```

To make the Poetry-managed kernel available for a globally installed Jupyter :
```bash
poetry run ipython kernel install --user --name=<KERNEL_NAME>
jupyter notebook
```
On the Jupyter server, select the created kernel in “Kernel” -> “Change kernel”.

### Dissecting Makefile
The Makefile provides several targets to assist in development and code quality :
- `init` creates a project-specific virtual environment and installs the dependencies of the .lock file.
- `ci` launches Black, Ruff, mypy and pytest on your source code.
- `pre-commit` set up and run pre-commit hooks (see pre-commit [documentation](https://pre-commit.com/)).
- `clean` clears bytecode, poetry/pip caches. Use with caution.


Requirements
-
- [Plotly](https://plotly.com/python/)
- [SciPy](https://scipy.org/)
- [Pydantic >=2.0](https://docs.pydantic.dev/)


Author
-
[Benjamin Roland](benjamin.roland@hotmail.fr)
