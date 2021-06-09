from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, analysis):
        self._analysis = analysis
        type = self._analysis.__class__.__name__
        if type == "MVPA":
            self.plot = self._plot_mvpa
        elif type == "RSA":
            self.plot = self._plot_rsa
        else:
            raise TypeError("Analysis type not handled.")

    def _plot_mvpa(self, show=False):
        fname = Path(__file__).parent.joinpath(
            "outputs",
            "plots",
            "mvpa",
            f"{self._analysis.feature.split('-')[1]}_{self._analysis.embedding.split('-')[1]}.png",
        )
        plt.hist(self._analysis.null, bins=25, color="turquoise", edgecolor="black")
        plt.axvline(self._analysis.score, color="black", linewidth=3)
        plt.xlim(
            {
                "task-code": [0.4, 0.9],
                "task-content": [0.4, 0.65],
                "task-structure": [0.25, 0.55],
                "code-bow": [0.4, 0.7],
                "code-tfidf": [0.4, 0.7],
            }[self._analysis.feature]
        )
        plt.savefig(fname)
        plt.show() if show else plt.clf()

    def _plot_rsa(self, show=False):
        fname = Path(__file__).parent.joinpath(
            "outputs", "plots", "rsa", f"{self._analysis.embedding.split('-')[1]}.jpg"
        )
        ticks = np.arange(self._analysis.corr.coef.shape[0])
        labels = np.array(["_".join(row) for row in self._analysis.corr.axes])
        indices = np.argsort(labels)
        plt.imshow(self._analysis.corr.coef[indices, :][:, indices])
        plt.xticks(ticks, labels[indices], fontsize=5, rotation=90)
        plt.yticks(ticks, labels[indices], fontsize=5)
        plt.clim([0, 1])
        plt.colorbar()
        plt.savefig(fname)
        plt.show() if show else plt.clf()
