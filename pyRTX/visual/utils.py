import matplotlib.colors as col


def color_convert(color, size = 1):
    color = col.to_rgba_array(color)*255.0
    return list(color)*size
