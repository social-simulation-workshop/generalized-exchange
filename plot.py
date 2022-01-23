import itertools
import os
import matplotlib.pyplot as plt
import numpy as np


class PlotLinesHandler(object):
    _ids = itertools.count(0)

    def __init__(self, xlabel, ylabel, ylabel_show, x_lim=None,
        figure_size=(15, 9), output_dir=os.path.join(os.getcwd(), "imgfiles")) -> None:
        super().__init__()

        self.id = next(self._ids)

        self.output_dir = output_dir
        self.title = "{}-{}".format(ylabel, xlabel)
        self.legend_list = list()

        plt.figure(self.id, figsize=figure_size, dpi=80)
        plt.title("{} - {}".format(ylabel_show, xlabel))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel_show)

        ax = plt.gca()
        ax.set_ylim([0., 10.])
        if x_lim is not None:
            ax.set_xlim([0, x_lim])

    def plot_line(self, data, legend,
        linewidth=1, color="", alpha=1.0):

        plt.figure(self.id)
        self.legend_list.append(legend)
        if color:
            plt.plot(np.arange(data.shape[-1]), data,
                linewidth=linewidth, color=color, alpha=alpha)
        else:
            plt.plot(np.arange(data.shape[-1]), data, linewidth=linewidth)

    def save_fig(self, title_param="", add_legend=True, title_lg=""):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        plt.figure(self.id)
        if add_legend:
            plt.legend(self.legend_list)
            title_lg = "_".join(self.legend_list)
        if title_param:
            fn = "_".join([self.title, title_lg, title_param]) + ".png"
        else:
            fn = "_".join([self.title, title_lg]) + ".png"
            
        plt.savefig(os.path.join(self.output_dir, fn))
        print("fig save to {}".format(os.path.join(self.output_dir, fn)))