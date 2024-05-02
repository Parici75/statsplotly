<div align="center" style="margin-bottom: 6vw;">

![Logo Dark](docs/assets/statsplotly-dark-mode-logo.png#gh-dark-mode-only)
![Logo Light](docs/assets/statsplotly-light-mode-logo.png#gh-light-mode-only)

</div>


[![Documentation](https://img.shields.io/website?label=docs&url=https://parici75.github.io/statsplotly)](https://parici75.github.io/statsplotly)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/statsplotly)
[![Black](https://img.shields.io/badge/Code%20style-Black-black)](https://black.readthedocs.io/en/stable/)
[![linting - Ruff](https://img.shields.io/badge/Linting-Ruff-yellow)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy.readthedocs.io/en/stable/index.html#)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![CI](https://github.com/Parici75/statsplotly/actions/workflows/test.yml/badge.svg)](https://github.com/Parici75/statsplotly/actions/workflows/test.yml)
[![PyPI - Package Version](https://img.shields.io/pypi/v/statsplotly)](https://pypi.org/project/statsplotly/)
[![GitHub License](https://img.shields.io/github/license/Parici75/statsplotly)](https://github.com/Parici75/statsplotly/blob/main/LICENSE)

----------------
[Statsplotly](https://github.com/parici75/statsplotly) is a Python data visualization library based on [Plotly](https://plotly.com/python/). It provides a high-level interface for drawing interactive statistical data visualization plots with a tidy, declarative API.


![statsplotly-demo](docs/assets/statsplotly-demo.gif)


## Philosophy

I began to work on this library during my first job outside academia, a few months before the fantastic [plotly.express](https://plotly.com/python/plotly-express/) API was launched. I was looking for a solution to generate visually appealing statistical plots that adapt well to diverse communication mediums, such as web applications, Jupyter Notebooks, presentations, and printed reports.

There are multiple declarative visualization library in the Python ecosystem, each with their own strengths, but none really fit my needs at the time.

Further, I discovered that real-world data analysis, whether in academia or industry, is rarely about creating facet plots. Instead, it revolves around crafting the most effective graphical representation to achieve one of two primary objectives:
  - **Exploratory** data visualization: the ideal tool should facilitate effortless immersion into the data to uncover valuable insights.
  - **Explanatory** data visualization: the ideal tool should enable rapid creation of polished, visually appealing plots that effectively support a hypothesis derived from the preceding analysis.

There is no reason these two endeavours should require different workflows. On the contrary, working towards both ends using the same tool should increase both analysis and communication efficiency.

I thus set out to design a library combining :
- a high level interface with graphical representation entities (e.g., plot, barplot, distplot, etc) as entrypoint functions. As intellectually satisfying the [Grammar of Graphics](https://www.tandfonline.com/doi/pdf/10.1198/jcgs.2009.07098) may be, I find this framework to be quite convoluted for expressing pragmatic visualization needs one has when performing real-world data analysis.

- a mainly declarative, yet chainable and customizable by imperative workflow, API. Assembling the appropriate data visualization units across a flexible combination of facets requires keeping control over the different layers of the graphical representations.

- A genuinely interactive plotting interface: In the realm of data visualization beyond scientific publishing, the primary objectives are to explore and convey insights about a phenomenon, business metrics, or a model's performance. This requires a responsive graphic engine allowing for dynamic visualization.


`statsplotly` key standout features are :
- an independent processing of color coding scheme, data slicer and plot dimensions.
- a high level interface for [seaborn-like](https://seaborn.pydata.org/tutorial/distributions.html) visualization of data distributions.
- statistical data processing under the hood.
- leveraging of the [tidy DataFrame](https://aeturrell.github.io/python4DS/data-tidy.html) structure for easy styling of plot cues (e.g., marker color, symbol, size, and opacity).
- sensible cartesian and coloraxis coordinates management across figure subplots.

In summary, `statsplotly` seeks to take advantage of the powerful interactivity offered by `plotly.js` without compromising statistical intelligibility for aesthetic choices, or vice-versa.


## Documentation

Main features and details of the public API can be found in the [documentation](https://parici75.github.io/statsplotly).


## Installation

### Using Pip

```bash
pip install statsplotly
```

## Development

### Using Poetry

First make sure you have Poetry installed on your system (see [instruction](https://python-poetry.org/docs/#installing-with-the-official-installer)).

Then, assuming you have a Unix shell with make, create and set up a new Poetry environment :

```bash
make init
```

To make the Poetry-managed kernel available for a globally installed Jupyter :

```bash
poetry run python -m ipykernel install --user --name=<KERNEL_NAME>
jupyter notebook
```

On the Jupyter server, select the created kernel in “Kernel” -> “Change kernel”.

### Dissecting Makefile

The Makefile provides several targets to assist in development and code quality :

- `init` creates a project-specific virtual environment and installs the dependencies of the `poetry.lock` file.
- `ci` launches Black, Ruff, mypy and pytest on your source code.
- `pre-commit` set up and run pre-commit hooks (see pre-commit [documentation](https://pre-commit.com/)).
- `update-doc` and `build-doc` generates documentation from source code and builds it with [Sphinx](https://www.sphinx-doc.org/en/master/index.html).
- `clean` clears bytecode, poetry/pip caches. Use with caution.

## Requirements

- [Plotly](https://plotly.com/python/)
- [SciPy](https://scipy.org/)
- [Pydantic >=2.0](https://docs.pydantic.dev/)

## Author

[Benjamin Roland](benjamin.roland@hotmail.fr)
