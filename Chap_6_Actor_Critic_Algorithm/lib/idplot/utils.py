def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

# m_supported_form = ['eps', 'jpeg', jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff]