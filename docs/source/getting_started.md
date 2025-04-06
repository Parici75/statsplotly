<div class="logo-container" style="width: 50%; margin: auto; margin-bottom: 1.9rem; margin-top: 1.9rem">
<picture>
  <img class="only-light" alt="Statsplotly Logo" src="_static/statsplotly-light-mode-logo.png">
  <img class="only-dark" alt="Statsplotly Logo" src="_static/statsplotly-dark-mode-logo.png">
</picture>
</div>

Statsplotly provides a high-level, declarative API for drawing statistical visualization with [plotly](https://plotly.com/).

Compared to the [plotly.express](https://plotly.com/python/plotly-express/) API, `statsplotly` color coding scheme, slicer and plot dimensions are independent.

This independence allows to leverage the powerful interactivity offered by [plotly.js](https://plotly.com/javascript/) without compromising statistical intelligibility for aesthetics choices, or vice-versa.

## Functions signature

All plotting functions return a {obj}`plotly.graph_objects.Figure` object, and must be supplied with :

- a {obj}`pandas.DataFrame`-comptatible data structure. Dictionaries or tidy (i.e., [long-form](https://en.wikipedia.org/wiki/Wide_and_narrow_data)) DataFrames are the recommended entry point. Note that hierarchical indexes are supported only on DataFrame's {obj}`~pandas.DataFrame.index`.
- column or index identifiers to specify `x`, `y` and `z` -when applicable- plotting dimensions.

> 💡 Read more on [tidy data](https://aeturrell.github.io/python4DS/data-tidy.html).

All plotting functions also accept:

- a `slicer` argument to slice the data along a particular dimension : each slice of the data is drawed as an independent `plotly.js` trace. Depending on the graphic representation, traces can be toggled via legend clicking, or dropdown selection.
- a `color_palette` argument which can be :

  - a string refering to a built-in [plotly](https://plotly.com/python/builtin-colorscales/), [seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html) or [matplotlib](https://matplotlib.org/stable/gallery/color/colormap_reference.html) colormap.
  - a list of CSS color names or HTML color codes:
    The color palette is used, by order of precedence :
    - To map color data specified by the `color` parameter onto the corresponding colormap.
    - to assign discrete colors to `slices` of data.

  String color data are interpreted as "discrete" (i.e., interval) colorscale. To specify a continuous colorscale, color data should be casted to numeric dtype.

- an `axis` argument to specify axes limits and aspect ratio. The `equal` and `square` styles of [Matlab API](https://fr.mathworks.com/help/matlab/ref/axis.html#buk989s-1-limits) are supported.
- a `title` argument to replace the default title assembled from the dimension names.
- a `fig`, `row`, `col` triplet argument to draw a subplot on a `Figure` object pre-declared with `plotly.subplots.make_subplots`.

Additional arguments can be provided, depending on the visualization selected.
