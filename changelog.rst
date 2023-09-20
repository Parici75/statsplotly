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
- Update `layout` and `marker` attributes to preserve each subplot original coloraxis upon adding new traces with `ColorAxis` specification on an existing `Figure`.

🎨 Feature improvements
- Improve consistency between axis range and axis format.
- Harmonize plot styling arguments.

💥 Breaking Changes
- Bump to Pydantic V2.

🚚 Rename package to `statsplotly`.

🚀 Set up Poetry management and Github Actions CI/CD.

🧹 Remove the `color` argument of `distplot` and `jointplot` functions because they require unidimensional trace color scheme.

➖ Remove dependency on `pymodules`.


Version 0.1.1
=============
**2022-03-10**

The 0.1.1 version fixes a bug in the :mod:`~statsplot.plot_specifiers.color` module.

🐛 Bug fixes
---------
- Bug in :obj:`ColorAxis` specification when using direct color assignments (i.e., CSS or hex color codes). The `colorscale` and `colorbar` attributes are now set to None.


Version 0.1.0
=============
**2023-02-15**

The 0.1.0 version is the first tagged release of the Statsplot package.
