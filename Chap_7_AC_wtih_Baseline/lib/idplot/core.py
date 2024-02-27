import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from itertools import cycle
import numpy as np

from .utils import cm2inch
from .config import default_cfg


def self_plot(data,
              fname=None,
              xlabel=None,
              ylabel=None,
              legend=None,
              legend_loc="best",
              color_list=None,
              xlim=None,
              ylim=None,
              xtick=None,
              ytick=None,
              yline=None,
              xline=None,
              usetex=False,
              ncol=1,
              figsize_scalar=1,
              display=True,
             ):
    """
    plot a single figure containing several curves.
    """

    # pre-process
    assert isinstance(data, (dict, list, tuple))

    if isinstance(data, dict):
        data = [data]
    num_data = len(data)

    fig_size = (default_cfg.fig_size * figsize_scalar, default_cfg.fig_size * figsize_scalar)
    _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg.dpi)

    # use tex to render fonts, tex install required
    if usetex:
        from matplotlib import rc
        rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
        rc('text', usetex=True)

    # color list
    if (color_list is None) or len(color_list) < num_data:
        tableau_colors = cycle(mcolors.TABLEAU_COLORS)
        color_list = [next(tableau_colors) for _ in range(num_data)]

    # plot figure
    for (i, d) in enumerate(data):
        plt.plot(d[0], d[1], color=color_list[i])

    # legend
    plt.tick_params(labelsize=default_cfg.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]

    if legend is not None:
        plt.legend(legend, loc=legend_loc, ncol=ncol, prop=default_cfg.legend_font)

    #  label
    plt.xlabel(xlabel, default_cfg.label_font)
    plt.ylabel(ylabel, default_cfg.label_font)

    if yline is not None:
        plt.axhline(yline, ls=":", c="grey")
    if xline is not None:
        plt.axvline(xline, ls=":", c="grey")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xtick is not None:
        plt.xticks(xtick)
    if ytick is not None:
        plt.yticks(ytick)
    plt.tight_layout(pad=default_cfg.pad)

    if fname is None:
        pass
    else:
        plt.savefig(fname)
    
    if display:
        plt.show()
    
    plt.close()

def self_plot_shadow(data,
              fname=None,
              xlabel=None,
              ylabel=None,
              legend=None,
              legend_loc="best",
              color_dark=None,
              color_light=None,
              xlim=None,
              ylim=None,
              xtick=None,
              ytick=None,
              yline=None,
              xline=None,
              usetex=False,
              ncol=1,
              figsize_scalar=1,
              display=True,
             ):
    """
    plot a single figure containing several curves.
    """

    # pre-process
    assert isinstance(data, (dict, list, tuple))

    if isinstance(data, dict):
        data = [data]
    num_data = len(data)

    fig_size = (default_cfg.fig_size * figsize_scalar, default_cfg.fig_size * figsize_scalar)
    _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg.dpi)

    # use tex to render fonts, tex install required
    if usetex:
        from matplotlib import rc
        rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
        rc('text', usetex=True)

    # color list
    if (color_dark is None) or len(color_dark) < num_data:
        tableau_colors = cycle(mcolors.TABLEAU_COLORS)
        color_dark = [next(tableau_colors) for _ in range(num_data)]
        color_light = [next(tableau_colors) for _ in range(num_data)]

    # plot figure
    for (i, d) in enumerate(data):
        x_data = d[0]
        d_array = np.array(d[1:])
        mean_data = np.array([np.mean(d_array[:,j]) for j in range(len(x_data))])
        plt.plot(x_data, mean_data, color=color_dark[i])
    
    # plot shadow
    for (i, d) in enumerate(data):
        x_data = d[0]
        d_array = np.array(d[1:])
        mean_data = np.array([np.mean(d_array[:,j]) for j in range(len(x_data))])
        std_data = np.array([np.std(d_array[:,j]) for j in range(len(x_data))])
        plt.fill_between(x_data, mean_data - std_data, mean_data + std_data, facecolor=color_light[i], alpha=0.3)

    
    # legend
    plt.tick_params(labelsize=default_cfg.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]

    if legend is not None:
        plt.legend(legend, loc=legend_loc, ncol=ncol, prop=default_cfg.legend_font)

    #  label
    plt.xlabel(xlabel, default_cfg.label_font)
    plt.ylabel(ylabel, default_cfg.label_font)

    if yline is not None:
        plt.axhline(yline, ls=":", c="grey")
    if xline is not None:
        plt.axvline(xline, ls=":", c="grey")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xtick is not None:
        plt.xticks(xtick)
    if ytick is not None:
        plt.yticks(ytick)
    plt.tight_layout(pad=default_cfg.pad)

    if fname is None:
        pass
    else:
        plt.savefig(fname)
    
    if display:
        plt.show()
    
    plt.close()