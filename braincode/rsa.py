from pathlib import Path

import numpy as np
from data import DataLoader
from mvpa import Analysis
from plots import Plotter


class RSA(Analysis):
    def __init__(self, network):
        super().__init__(network)
        self._corr = CorrelationMatrix(self.network)

    @property
    def corr(self):
        return self._corr

    def _calc_corr(self):
        for subject in sorted(self.corr.loader.datadir.iterdir()):
            self.corr.add_subject(subject)

    def run(self):
        self._calc_corr()
        self._plot()
        return self


class CorrelationMatrix:
    def __init__(self, network):
        self._network = network
        self._loader = DataLoader(self._network)
        self._matrix = np.zeros((self.loader.samples, self.loader.samples))
        self._axes = np.array([])
        self._subjects = 0

    @property
    def loader(self):
        return self._loader

    @property
    def axes(self):
        return self._axes

    @property
    def coef(self):
        return self._matrix / self._subjects

    def _update_coef(self, data):
        self._matrix += np.corrcoef(data)
        self._subjects += 1

    def add_subject(self, subject):
        X, content, lang, structure = self.loader.get_xcls(subject)
        self._update_coef(X)
        if not self.axes.size:
            self._axes = np.vstack(
                [self.loader.formatcell(arr) for arr in [content, lang, structure]]
            ).T
