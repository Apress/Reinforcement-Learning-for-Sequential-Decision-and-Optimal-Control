import os
import matplotlib.pyplot as plt
from .core import self_plot


def plot_env(save_path,
             t_a=None,
             t_v=None,
             t_d=None,
             t_turning=None,
             fig_form="png"):
    assert fig_form in plt.gcf().canvas.get_supported_filetypes(), "Figure form not supported, try pdf, svg, png, jpg"

    if t_a is not None:
        path_ = os.path.join(save_path, "t_a." + fig_form)
        self_plot(t_a,
                    path_,
                    xlabel="$t\ \mathregular{[s]}$",
                    ylabel="$a\ \mathregular{[m/s^2]}$")

    if t_v is not None:
        path_ = os.path.join(save_path, "t_v." + fig_form)
        self_plot(t_v,
                    path_,
                    xlabel="$t\ \mathregular{[s]}$",
                    ylabel="$v\ \mathregular{[m/s]}$")

    if t_d is not None:
        path_ = os.path.join(save_path, "t_d." + fig_form)
        self_plot(t_d,
                    path_,
                    xlabel="$t\ \mathregular{[s]}$",
                    ylabel="$d\ \mathregular{[km]}$")

    if t_turning is not None:
        path_ = os.path.join(save_path, "t_turning." + fig_form)
        self_plot(t_turning,
                    path_,
                    xlabel="$t\ \mathregular{[s]}$",
                    ylabel="$\delta \ \mathregular{[rad]}$")