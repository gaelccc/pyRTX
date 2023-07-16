import matplotlib.colors as col


def color_convert(color, size = 1):
    """Converts a color to a list of colors of size `size`
    Parameters
    ----------
    color : str or tuple
        Color to be converted
    size : int
        Size of the list of colors to be returned
    Returns
    -------
    color : (size,) array-like
        converted color as a list of rgba lists
            """
    color = col.to_rgba_array(color)*255.0
    return list(color)*size
