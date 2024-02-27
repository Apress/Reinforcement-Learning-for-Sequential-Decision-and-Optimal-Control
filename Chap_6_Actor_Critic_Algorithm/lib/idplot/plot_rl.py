import os
import matplotlib.pyplot as plt
from .core import self_plot


def plot_rl(save_path,
            iter_tar=None,
            sample_tar=None,
            iter_walltime=None,
            sample_walltime=None,
            fig_form="png"):
    assert fig_form in plt.gcf().canvas.get_supported_filetypes(), "Figure form not supported, try pdf, svg, png, jpg"

    if iter_tar is not None:
        path_ = os.path.join(save_path, "iteration_tar." + fig_form)
        self_plot(iter_tar,
                    path_,
                    xlabel="Iteration",
                    ylabel="Total Average Return")

    if sample_tar is not None:
        path_ = os.path.join(save_path, "sample_tar." + fig_form)
        self_plot(sample_tar,
                    path_,
                    xlabel="Sample",
                    ylabel="Total Average Return")

    if iter_walltime is not None:
        path_ = os.path.join(save_path, "iteration_walltime." + fig_form)
        self_plot(iter_walltime,
                    path_,
                    xlabel="Iteration",
                    ylabel="Wall Clock Time")

    if sample_walltime is not None:
        path_ = os.path.join(save_path, "sample_walltime." + fig_form)
        self_plot(sample_walltime,
                    path_,
                    xlabel="Sample",
                    ylabel="Wall Clock Time")