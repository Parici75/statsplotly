Version 0.1.2
=============
**2023-07-11**

The 0.1.2 version allows to combine several subplots with different :obj:`ColorAxis` on the same figure, and improve axes management.

Feature improvements
--------------------
- Update `layout` and `marker` attributes to preserve each subplot original coloraxis upon adding new traces with :obj:`ColorAxis` specification on an existing :obj:`Figure`.
- Improve consistency between axis range and axis format.
- Harmonize plot styling arguments.

Cleaning
--------
- Remove the `color` argument of :func:`distplot` and :func:`jointplot` functions because they require unidimensional trace color scheme.
- Bump to Pydantic V2.
- Remove dependency on :mod:`pymodules`


Version 0.1.1
=============
**2022-03-10**

The 0.1.1 version fixes a bug in the :mod:`~statsplot.plot_specifiers.color` module.

Bug fixes
---------
- Bug in :obj:`ColorAxis` specification when using direct color assignments (i.e., CSS or hex color codes). The `colorscale` and `colorbar` attributes are now set to None.


Version 0.1.0
=============
**2023-02-15**

The 0.1.0 version is the first tagged release of the Statsplot package.
