import numpy as np
import pandas as pd
# can be deleted if tk is installed
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import StrMethodFormatter

from matplotlib.colors import LogNorm

from tensorsignatures.config import *

class MidPointLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=1, clip=False):
        LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.array(np.interp(np.log(value), x, y), mask=result.mask, copy=False)

def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))


def set_grid_title(G, title, color, size=10, va='center', ha='center',
                   rotation=0):

    ax = plt.subplot(G)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.text(0.5, 0.5, title, color=color, ha=ha, va=va, size=size, alpha=1,
             rotation=rotation)


def heatmap(data, row_labels=None, col_labels=None, ax=None, vmin=None,
            vmax=None, cbar_ticks=None, cbar_aspect=10, cbarlabel="",
            annotate=True, cmap='RdBu', **kwargs):
    """Create a heatmap from a numpy array and two lists of labels. Adapted from
    https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Args:
        data (:obj:`array`): A 2D numpy array of shape (N, M).
        row_labels (:obj:`list`): A list or array of length N with the labels
            for the rows.
        col_labels (:obj:`list`): A list or array of length M with the labels
            for the columns.
        ax: A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
            If not provided, use current axes or create a new one. Optional.
        cbar_kw (:obj:`dict`): A dictionary with arguments to
            :obj:`matplotlib.Figure.colorbar`. Optional.
        cbarlabel (:obj:`str`): The label for the colorbar. Optional.
        **kwargs: All other arguments are forwarded to `imshow`.
    Returns:
        The axes and colorbar object of the heatmap
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data,
                   norm=MidPointLogNorm(vmin, vmax),
                   interpolation='none',
                   aspect='auto',
                   cmap=cmap,
                   **kwargs)

    # Create colorbar
    if cbar_ticks is None:
        cbar_ticks = [j * 10**i for i in range(-3, 2) for j in [1, 2, 4, 6, 8]]
    cbar = ax.figure.colorbar(im,
                              ax=ax,
                              aspect=cbar_aspect,
                              format=FuncFormatter(ticks_format),
                              ticks=cbar_ticks)

    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if col_labels is None:
        col_labels = [str(i) for i in range(data.shape[1])]
    ax.set_xticklabels(col_labels)
    if row_labels is None:
        row_labels = [str(i) for i in range(data.shape[0])]
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, right=False,
                   labeltop=True, labelbottom=False)
    ax.tick_params(which='major', axis='both', direction='out', length=3,
                   width=1)
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False, right=False,
                   top=False)

    if annotate:
        annotate_heatmap(im)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap. Adapted from
    https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Args:
        im: The AxesImage to be labeled.
        data: Data used to annotate.  If None, the image's data is used.
        Optional.
    valfmt: The format of the annotations inside the heatmap.  This should
        either use the string format method, e.g. "$ {x:.2f}", or be a
        :obj:`matplotlib.ticker.Formatter`. Optional.
    textcolors: A list or array of two color specifications.  The first is used
        for values below a threshold, the second for those above. Optional.
    threshold: Value in data units according to which the colors from
        textcolors are applied. If None (the default) uses the middle of the
        colormap as separation. Optional.
    **kwargs: All other arguments are forwarded to each call to `text` used to
        create the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.
        min_val, max_val = im.get_clim()
        low_cut = np.median(np.logspace(np.log10(min_val), 0))
        up_cut = np.median(np.logspace(0, np.log10(max_val)))


    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] < min_val or data[i, j] > max_val:
                continue
            idx=0
            if data[i, j] > up_cut or data[i, j] < low_cut:
                idx=1
            #kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            kw.update(color=textcolors[idx])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_signature(signature_tensor,
                   signature,
                   file_path=None,
                   width=0.4,
                   fig=None):
    r"""Plots a single signature.

    Args:
        signature_tensor (:obj:`array`, shape :math:`(3, 3, -1, p, s)`):
            Signature tensor in which generic genomic dimensions are collapsed.
        signature (:obj:`int`, :math:`0 \leq i < s`): Signature i that shall be
            plotted.
        file_path (:obj:`str`): If provided function will save the plot to
            disk.
    Returns:
        :obj:`list`: Returns a list axes.

    Example:

    In some cases it may be necessary reshape the signature tensor.

    >>> from tensorsignature.plot import plot_signatures
    >>> S.shape
    (3,3,16,4,2,96,6)
    >>> ts.plot_signature(S.reshape(3,3,-1,96,6), 0) # plots first signature
    """
    if fig is None:
        fig = plt.gcf()
    G = gs.GridSpec(8, 12)
    G.update(wspace=0.06, hspace=0.05)
    dx = 16

    ax_limits = []
    ax_list = []

    set_grid_title(G[0, 0:6],
                   'Transcription (template strand light, coding strand dark)',
                   'black', size=12)
    for i, j in enumerate(range(0, dx * 6, dx)):
        ax = fig.add_subplot(G[2:, i:i + 1])
        ax.bar(np.arange(dx),
               signature_tensor[0, 2, 0, j:j + dx, signature].reshape(-1),
               color=COLORPAIRS[i][1],
               width=width,
               edgecolor="none")
        ax.bar(np.arange(dx) + width,
               signature_tensor[1, 2, 0, j:j + dx, signature].reshape(-1),
               color=COLORPAIRS[i][0],
               width=width,
               edgecolor="none")

        if i == 0:
            ax.tick_params(which='major',
                           direction='out',
                           length=0,
                           width=1,
                           axis='y')
            ax.tick_params(which='minor',
                           direction='out',
                           length=0,
                           width=1,
                           axis='y')

        if i >= 1:
            ax.spines['left'].set_visible(False)
            ax.set_yticklabels([])
            ax.tick_params(which='major',
                           direction='out',
                           length=0,
                           width=1,
                           axis='y')
            ax.tick_params(which='minor',
                           direction='out',
                           length=0,
                           width=1,
                           axis='y')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(which='both',
                       direction='out',
                       length=3,
                       width=1,
                       axis='x')
        ax.set_xticks(np.arange(dx) + width / 2)
        ax.set_xticklabels(SNV_MUT[j:j + dx], rotation=90, fontsize=8)

        ax_limits.extend([*ax.get_ylim()])
        ax_list.append(ax)

        ax_title = fig.add_subplot(G[1, i:i + 1])
        set_grid_title(ax_title, SNV_MUT_TYPES[i - 6], COLORS[i - 6], size=12)

    set_grid_title(G[0, 6:],
                   'Replication (lagging strand light, leading strand dark)',
                   'black',
                   size=12)
    for i, j in enumerate(range(0, dx * 6, dx)):
        i += 6

        ax = fig.add_subplot(G[2:, i:i + 1])
        ax.bar(np.arange(dx),
               signature_tensor[2, 0, 0, j:j + dx, signature].reshape(-1),
               # yerr=BOOTSTRAP['S'][2, 0, 0, j:j + dx, signature].T,
               color=COLORPAIRS[i - 6][1],
               width=width,
               edgecolor="none")
        ax.bar(np.arange(dx) + width,
               signature_tensor[2, 1, 0, j:j + dx, signature].reshape(-1),
               # yerr=BOOTSTRAP['S'][2, 1, 0, j:j + dx, signature].T,
               color=COLORPAIRS[i - 6][0],
               width=width,
               edgecolor="none")
        # ax.set_xticks(np.arange(dx))

        if i == 6:
            ax.set_yticklabels([])
            ax.tick_params(which='major',
                           direction='out',
                           length=0,
                           width=1,
                           axis='y')
            ax.tick_params(which='minor',
                           direction='out',
                           length=0,
                           width=1,
                           axis='y')

        if i >= 7:
            ax.spines['left'].set_visible(False)
            ax.set_yticklabels([])
            ax.tick_params(which='major',
                           direction='out',
                           length=0,
                           width=1,
                           axis='y')
            ax.tick_params(which='minor',
                           direction='out',
                           length=0,
                           width=1,
                           axis='y')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(which='both', direction='out',
                       length=3, width=1, axis='x')
        ax.set_xticks(np.arange(dx) + width / 2)

        ax.set_xticklabels(SNV_MUT[j:j + dx], rotation=90, fontsize=8)
        ax_limits.extend([*ax.get_ylim()])
        ax_list.append(ax)

        ax_title = fig.add_subplot(G[1, i:i + 1])
        set_grid_title(ax_title, SNV_MUT_TYPES[i - 6], COLORS[i - 6], size=12)

    ymin, ymax = np.min(ax_limits), np.max(ax_limits)
    for axi in ax_list:
        axi.set_ylim(top=ymax)

        axi.yaxis.grid(True)
        axi.set_axisbelow(True)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_signatures(signature_tensor, bootstrap=None, width=0.3, fig=None):
    """Convinience function to plot all singate.

    Args:
        signature_tensor (:obj:`array`, shape (3, 3, -1, p, n)): Signature
            tensor.
        width (:float:`int`): Width of the bars.
        fig (:obj:`matplotlib.figure.Figure`): A matpltotlib figure object, if
            no figure is provided.
    Results:
        A :obj:`list` containing all axes.

    Examples:

    >>> import tensorsignature as ts
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize = (16, 4))
    >>> plot_signatures(S)

    """
    if fig is None:
        fig = plt.gcf()

    outer = gs.GridSpec(signature_tensor.shape[-1], 2, wspace=0.01)

    for s in range(signature_tensor.shape[-1]):
        inner_trx = gs.GridSpecFromSubplotSpec(
            1, 6, subplot_spec=outer[s, 0], wspace=0.05)
        inner_rep = gs.GridSpecFromSubplotSpec(
            1, 6, subplot_spec=outer[s, 1], wspace=0.05)
        dx = 16
        ax_limits = []
        ax_list = []

        for i, j in enumerate(range(0, dx * 6, dx)):
            axt = fig.add_subplot(inner_trx[:, i:i + 1])
            axr = fig.add_subplot(inner_rep[:, i:i + 1])

            if bootstrap is not None:
                yerr = bootstrap.yerr('S').reshape(
                    3, 3, -1, signature_tensor.shape[-2],
                    signature_tensor.shape[-1], 2)
                yerr_tc = yerr[0, 2, 0, j:j + dx, s, :].T
                yerr_tt = yerr[1, 2, 0, j:j + dx, s, :].T
                yerr_rl = yerr[2, 0, 0, j:j + dx, s, :].T
                yerr_rg = yerr[2, 1, 0, j:j + dx, s, :].T
            else:
                yerr_tc = None
                yerr_tt = None
                yerr_rl = None
                yerr_rg = None

            axt.bar(np.arange(dx),
                    signature_tensor[0, 2, 0, j:j + dx, s].reshape(-1),
                    yerr=yerr_tc,
                    color=COLORPAIRS[i][1],
                    width=width,
                    edgecolor="none")
            axt.bar(np.arange(dx) + width,
                    signature_tensor[1, 2, 0, j:j + dx, s].reshape(-1),
                    yerr=yerr_tt,
                    color=COLORPAIRS[i][0],
                    width=width,
                    edgecolor="none")

            axr.bar(np.arange(dx),
                    signature_tensor[2, 0, 0, j:j + dx, s].reshape(-1),
                    yerr=yerr_rl,
                    color=COLORPAIRS[i][1],
                    width=width,
                    edgecolor="none")
            axr.bar(np.arange(dx) + width,
                    signature_tensor[2, 1, 0, j:j + dx, s].reshape(-1),
                    yerr=yerr_rg,
                    color=COLORPAIRS[i][0],
                    width=width,
                    edgecolor="none")

            if i == 0:
                axt.tick_params(which='both',
                                direction='out',
                                length=3,
                                width=1,
                                axis='y',
                                right=False)
                axt.set_ylabel(s)
                axr.tick_params(which='both',
                                direction='out',
                                length=3,
                                width=1,
                                axis='y',
                                right=False)
                axr.set_yticklabels([])

            if i >= 1:
                axt.spines['left'].set_visible(False)
                axt.set_yticklabels([])
                axt.tick_params(which='major',
                                direction='out',
                                length=0,
                                width=1,
                                axis='y')
                axt.tick_params(which='minor',
                                direction='out',
                                length=0,
                                width=1,
                                axis='y')

                axr.spines['left'].set_visible(False)
                axr.set_yticklabels([])
                axr.tick_params(which='major',
                                direction='out',
                                length=0,
                                width=1,
                                axis='y')
                axr.tick_params(which='minor',
                                direction='out',
                                length=0,
                                width=1,
                                axis='y')

            axt.spines['top'].set_visible(False)
            axt.spines['right'].set_visible(False)
            axr.spines['top'].set_visible(False)
            axr.spines['right'].set_visible(False)

            axt.tick_params(which='both',
                            direction='out',
                            length=3,
                            width=1,
                            axis='x',
                            top=False,
                            labeltop=False)
            axr.tick_params(which='both',
                            direction='out',
                            length=3,
                            width=1,
                            axis='x',
                            top=False,
                            labeltop=False)
            axt.set_xticks(np.arange(dx) + width)
            axr.set_xticks(np.arange(dx) + width)
            axt.set_xticklabels([])
            axr.set_xticklabels([])

            ax_limits.extend([*axt.get_ylim()])
            ax_limits.extend([*axr.get_ylim()])
            ax_list.append(axt)
            ax_list.append(axr)


            # ax_title = fig.add_subplot(G[1, i:i + 1])
            # set_title(ax_title, SNV_MUT_TYPES[i - 6], COLORS[i - 6], size=12)

        ymin, ymax = np.min(ax_limits), np.max(ax_limits)
        for axi in ax_list:
            axi.set_ylim(top=ymax)
            axi.yaxis.set_major_locator(plt.MaxNLocator(4))
            #plt.locator_params(axis='y', nbins=6)


            axi.yaxis.grid(True)
            axi.set_axisbelow(True)
