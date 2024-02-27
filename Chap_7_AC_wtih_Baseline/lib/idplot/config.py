

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


default_cfg = AttrDict()

default_cfg.fig_size = (8.5, 6.5)
default_cfg.dpi = 600
default_cfg.pad = 0.2

default_cfg.tick_size = 8
default_cfg.tick_label_font = 'Times New Roman'
default_cfg.legend_font = {'family': 'Times New Roman', 'size': '8', 'weight': 'normal'}
default_cfg.label_font = {'family': 'Times New Roman', 'size': '9', 'weight': 'normal'}