from pathlib import Path

import numpy as np
from data import DataLoader
from decoding import Analysis
from plots import Plotter


class RSA(Analysis):
    def __init__(self, feature, base_path):
        super().__init__(feature, base_path)

    def run(self):
        self.rdm = RDM(self.feature, self._base_path).run()
        return self


class RDM(Analysis):
    def __init__(self, feature, base_path):
        super().__init__(feature, base_path)
        self._loader = DataLoader(self._base_path, self._feature)
        self._matrix = np.zeros((self._loader.samples, self._loader.samples))
        self._axes = np.array([])
        self._subjects = 0

    @property
    def axes(self):
        if self._axes.size == 0:
            raise RuntimeError("Axes not set. Need to add subject.")
        return self._axes

    @property
    def coef(self):
        if self._subjects == 0:
            raise RuntimeError("Coefficients not set. Need to add subject.")
        return 1 - (self._matrix / self._subjects)

    def _update_coef(self, data):
        self._matrix += np.corrcoef(data)
        self._subjects += 1

    def _add_subject(self, subject):
        X, axes = self._loader.get_data_rsa(subject)
        self._update_coef(X)
        if not self._axes.size:
            self._axes = axes

    def _calc_corr(self):
        subjects = sorted(self._loader.datadir.joinpath("neural_data").glob("*.mat"))
        for subject in subjects:
            self._add_subject(subject)

    def run(self):
        self._calc_corr()
        self._plot()
        return self
