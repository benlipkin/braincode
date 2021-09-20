import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, analysis):
        self._analysis = analysis
        self._feature = self._analysis.feature.split("-")[1]
        self._target = self._analysis.target.split("-")[1]
        self._type = self._analysis.__class__.__name__
        if self._type in ["MVPA", "PRDA", "RSA"]:
            self.plot = self._plot_decoder
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
        plt.xlim({"MVPA": [0.25, 0.75], "PRDA": [0, 1], "RSA": [0, 0.1]}[self._type])
        plt.savefig(fname)
        plt.show() if show else plt.clf()
        self._logger.info(f"Plotting '{fname.name}'.")
