import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import gs

class MidPointLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.array(np.interp(np.log(value), x, y), mask=result.mask, copy=False)


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

def plot_signature(signature_tensor,
                   signature,
                   file_path=None,
                   figsize=(16, 2)):
    r"""Plots signatures.

    Args:
        signature_tensor (:obj:`array`, shape :math:`(3, 3, -1, p, s)`):
            Signature tensor in which generic genomic dimensions are collapsed.
        signature (:obj:`int`, :math:`0 \leq i < s`): Signature i that shall be
            plotted.
        file_path (:obj:`str`): If provided function will save the plot to
            disk.

    """
    fig = plt.figure(figsize=figsize, facecolor='white')  # 12, 6
    G = gs.GridSpec(8, 12)
    G.update(wspace=0.06, hspace=0.05)
    dx = 16

    ax_limits = []
    ax_list = []
    width = 0.3

    set_grid_title(G[0, 0:6], 'Transcription (template strand light, coding strand dark)', 'black', size=12)
    for i, j in enumerate(range(0, dx * 6, dx)):
        ax = fig.add_subplot(G[2:, i:i + 1])
        print(COLORPAIRS[i][1])
        ax.bar(np.arange(dx), signature_tensor[0, 2, 0, j:j + dx, signature].reshape(-1),
               color=COLORPAIRS[i][1],
               width=width,edgecolor = "none")
        ax.bar(np.arange(dx)+width, signature_tensor[1, 2, 0, j:j + dx, signature].reshape(-1),
               color=COLORPAIRS[i][0],
               width=width, edgecolor = "none")
        if i==0:
            ax.tick_params(which='major', direction='out', length=0, width=1, axis='y')
            ax.tick_params(which='minor', direction='out', length=0, width=1, axis='y')

        if i>=1:
            ax.spines['left'].set_visible(False)
            ax.set_yticklabels([])
            ax.tick_params(which='major', direction='out', length=0, width=1, axis='y')
            ax.tick_params(which='minor', direction='out', length=0, width=1, axis='y')
            #ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(which='both', direction='out', length=3, width=1, axis='x')
        ax.set_xticks(np.arange(dx)+width/2)
        ax.set_xticklabels(SNV_MUT[j:j + dx], rotation=90, fontsize=8)

        ax_limits.extend([*ax.get_ylim()])
        ax_list.append(ax)


        ax_title = fig.add_subplot(G[1, i:i + 1])
        set_grid_title(ax_title, SNV_MUT_TYPES[i - 6], COLORS[i - 6], size=12)

    set_grid_title(G[0, 6:], 'Replication (lagging strand light, leading strand dark)', 'black', size=12)
    for i, j in enumerate(range(0, dx * 6, dx)):
        i += 6

        ax = fig.add_subplot(G[2:, i:i + 1])
        ax.bar(np.arange(dx), signature_tensor[2, 0, 0, j:j + dx, signature].reshape(-1),
               #yerr=BOOTSTRAP['S'][2, 0, 0, j:j + dx, signature].T,
               color=COLORPAIRS[i-6][1],
               width=width, edgecolor = "none"
              )
        ax.bar(np.arange(dx)+width, signature_tensor[2, 1, 0, j:j + dx, signature].reshape(-1),
               #yerr=BOOTSTRAP['S'][2, 1, 0, j:j + dx, signature].T,
               color=COLORPAIRS[i-6][0],
               width=width, edgecolor = "none")
        # ax.set_xticks(np.arange(dx))

        if i==6:
            ax.set_yticklabels([])
            ax.tick_params(which='major', direction='out', length=0, width=1, axis='y')
            ax.tick_params(which='minor', direction='out', length=0, width=1, axis='y')

        if i>=7:
            ax.spines['left'].set_visible(False)
            ax.set_yticklabels([])
            ax.tick_params(which='major', direction='out', length=0, width=1, axis='y')
            ax.tick_params(which='minor', direction='out', length=0, width=1, axis='y')
            #ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(which='both', direction='out', length=3, width=1, axis='x')
        ax.set_xticks(np.arange(dx)+width/2)

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
        plt.savefig(file_path.rstrip('.png') + '.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
