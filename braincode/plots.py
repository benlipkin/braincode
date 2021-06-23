import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, analysis):
        self._analysis = analysis
        self._feature = self._analysis.feature.split("-")[1]
        self._type = self._analysis.__class__.__name__
        if self._type in ["MVPA", "PRDA"]:
            self._target = self._analysis.target.split("-")[1]
            self.plot = self._plot_decoder
        elif self._type == "RSA":
            self.plot = self._plot_rsa
        else:
            raise TypeError("Analysis type not handled.")
        self._logger = logging.getLogger(self.__class__.__name__)

    def _plot_decoder(self, show=False):
        fname = Path(__file__).parent.joinpath(
            "outputs",
            "plots",
            self._type.lower(),
            f"{self._feature}_{self._target}.png",
        )
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
        plt.hist(self._analysis.null, bins=25, color="turquoise", edgecolor="black")
        plt.axvline(self._analysis.score, color="black", linewidth=3)
        plt.xlim([0, 1])
        plt.savefig(fname)
        plt.show() if show else plt.clf()
        self._logger.info(f"Plotting {fname.name}")

    def _plot_rsa(self, show=False):
        fname = Path(__file__).parent.joinpath(
            "outputs", "plots", self._type.lower(), f"{self._feature}.jpg"
        )
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
        ticks = np.arange(self._analysis.coef.shape[0])
        labels = np.array(["_".join(row) for row in self._analysis.axes])
        indices = np.argsort(labels)
        plt.imshow(self._analysis.coef[indices, :][:, indices])
        plt.xticks(ticks, labels[indices], fontsize=5, rotation=90)
        plt.yticks(ticks, labels[indices], fontsize=5)
        plt.clim([0, 1])
        plt.colorbar()
        plt.savefig(fname)
        plt.show() if show else plt.clf()
        self._logger.info(f"Plotting {fname.name}")
