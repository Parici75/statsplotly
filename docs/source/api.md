## Interactive statistical data visualization with Statsplot

Compared to `plotly.express` API, `statsplot` color coding scheme, slicer and plot dimensions are independent.

This independence allows to leverage the powerful interactivity offered by plotly.js without compromising statistical intelligibility for aesthetics choices, or vice-versa.


### Functions signature

All functions return a `plotly.graphic_objects.Figure` object.

All plotting functions must be supplied with :

   - a tidy (i.e., [long-form](https://en.wikipedia.org/wiki/Wide_and_narrow_data)) DataFrame argument for data, with a flat header (hierarchical index are supported).
   - column identifiers to specify plotting dimensions.

All plotting functions also accept:
   - a `slicer` argument to slice the data along a particular dimension. Depending on the graphic representation, slices of data are plotted as independent traces (scatter-like representation), or as dropdown items (heatmap).
   - a `color_palette` argument which can be :
        - a string refering to a built-in `plotly`, `seaborn` or `matplotlib` colormap.
        - a `List` of CSS color names or HTML color codes.
        The color palette is used, by order of precedence :
            - to map color data of the `color` parameter onto colors.
            - to assign discrete colors to `slices` of data.

    String color data are interpreted as "discrete" (i.e., interval) colorscale. To specify a continuous colorscale, color data should be cast to numeric type.


   - an `axis` argument to specify axes limits and aspect ratio. The `equal` and `square` styles of [Matlab API](https://fr.mathworks.com/help/matlab/ref/axis.html#buk989s-1-limits) are supported.
   - a `fig`, `row`, `col` triplet argument to draw on a pre-declared `Figure`object.

Additional arguments depends on the visualisation selected.
