Version 0.2.0
=============
**2024-04-05**

The 0.2.0 version introduces new methods enhancing subplots management, and improves plotting features.

✨ New features
***************
- Add the :class:`~statsplotly.utils.SubplotGridFormatter` class, exposing methods to manage the subplot grid :
    - :meth:`~statsplotly.utils.SubplotGridFormatter.set_common_coloraxis` manages coloraxis and colorbar display across multiple heatmap subplots.
    - :meth:`~statsplotly.utils.SubplotGridFormatter.set_common_axis_limit` manages axis limits across a subplot grid.
    - :meth:`~statsplotly.utils.SubplotGridFormatter.tidy_subplots` manages post-processing of a subplot grid.

- Enhance the color specification option of the main plotting module : the `color` argument now accepts reference to `datetime` and arbitrary `string` color data.
- Support 'geometric mean' and 'geometric standard deviation factor' options in :func:`~statsplotly.barplot` `aggregation_fct` and `error_bar` arguments.

🎨 Feature improvements
***********************
- Improve :func:`~statsplotly.heatmap` axis formatting when `axis = "equal"`.
- Improve :func:`~statsplotly.jointplot` trace visibility groups.
- Add `color` specification option for :func:`~statsplotly.jointplot` scatter traces.
- Preserve original data ordering in :func:`~statsplotly.barplot` aggregations.
- Improve colorbar management in layout.
- Improve `datetime` data handling in :func:`~statsplotly.catplot`.
- Improve step histogram line aesthetics.
- Display slice name on color-coded barcharts.

🐛 Bug fixes
************
- Fix data slice visibility inconsistencies in :func:`~statsplotly.heatmap`.
- Fix `scaleratio` for `axis="id_line"`.
- Fix cumulative histograms when `cumulative=True` and `step=True`.

Version 0.1.5
=============
**2023-11-25**

The 0.1.5 version ensures compatibility with Python>=3.10.


Version 0.1.4
=============
**2023-09-27**

The 0.1.4 version fix inconsistencies in color coding.

🐛 Make casting color coding string array to integer consistent.

🎨 Add support for all Plotly / Seaborn and Matplotlib colorscales.

💥 Bump to pydantic==v2.4


Version 0.1.3
=============
**2023-09-20**

The 0.1.3 version fixes a bug when processing color coding arrays.

🐛 Handle `null` values when casting color coding array to integer data type.

💚 Set up pre-commit hooks.

🔨 Clean up deprecated Pandas code.

👕 Linting


Version 0.1.2
=============
**2023-07-11**

The 0.1.2 version allows to combine several subplots with different `ColorAxis` on the same figure, and improve axes management.

✨ New features
***************
- Update `layout` and `marker` attributes to preserve each subplot original coloraxis upon adding new traces with `ColorAxis` specification on an existing `Figure`.

🎨 Feature improvements
***********************
- Improve consistency between axis range and axis format.
- Harmonize plot styling arguments.

💥 Breaking Changes
*******************
- Bump to Pydantic V2.

🚚 Rename package to `statsplotly`.

🚀 Set up Poetry management and Github Actions CI/CD.

🧹 Remove the `color` argument of `distplot` and `jointplot` functions because they require unidimensional trace color scheme.

➖ Remove dependency on `pymodules`.


Version 0.1.1
=============
**2022-03-10**

The 0.1.1 version fixes a bug in the :mod:`~statsplot.plot_specifiers.color` module.

🐛 Fix bug in :obj:`ColorAxis` specification when using direct color assignments (i.e., CSS or hex color codes). The `colorscale` and `colorbar` attributes are now set to None.


Version 0.1.0
=============
**2023-02-15**

The 0.1.0 version is the first tagged release of the Statsplot package.
