import matplotlib.pyplot as plt
from datetime import datetime
import textwrap

import numpy as np


def plot_train_loss(hp_sentence, viz):
    """
    Visualize train & validation loss & metrics. hp_sentence is used as the title of the plot.

    Saves plots in the plots folder.
    """
    if 'val_loss_list' in viz.keys():
        fig = plt.figure()
        x = np.arange(len(viz['train_loss_list']))
        plt.title('\n'.join(textwrap.wrap(hp_sentence, 60)))
        fig.tight_layout()
        plt.rcParams["axes.titlesize"] = 6
        plt.plot(x, viz['train_loss_list'])
        plt.plot(x, viz['val_loss_list'])
        plt.legend(['training loss', 'valid loss'], loc='upper left')
        plt.savefig('plots/' + str(datetime.now())[:-10] + 'loss.png')
        plt.close(fig)

    if 'val_recall_list' in viz.keys():
        fig = plt.figure()
        x = np.arange(len(viz['train_precision_list']))
        plt.title('\n'.join(textwrap.wrap(hp_sentence, 60)))
        fig.tight_layout()
        plt.rcParams["axes.titlesize"] = 6
        plt.plot(x, viz['train_precision_list'])
        plt.plot(x, viz['train_recall_list'])
        plt.plot(x, viz['train_coverage_list'])
        plt.plot(x, viz['val_precision_list'])
        plt.plot(x, viz['val_recall_list'])
        plt.plot(x, viz['val_coverage_list'])
        plt.legend(['training precision', 'training recall', 'training coverage/10',
                    'valid precision', 'valid recall', 'valid coverage/10'], loc='upper left')
        plt.savefig('plots/' + str(datetime.now())[:-10] + 'metrics.png')
        plt.close(fig)
