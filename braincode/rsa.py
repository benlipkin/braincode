from pathlib import Path

import numpy as np
from data import DataLoader
from decoding import Analysis
from plots import Plotter


class RSA(Analysis):
    def __init__(self, feature, target, base_path):
        super().__init__(feature, target, base_path)

    @staticmethod
    def _calc_rsa(brain_rdm, model_rdm):
        if not brain_rdm.matrix.size == model_rdm.matrix.size:
            raise RuntimeError("RDMs mismatched. Check feature target pair.")
        indices = np.triu_indices(brain_rdm.matrix.shape[0], k=1)
        return np.corrcoef(brain_rdm.matrix[indices], model_rdm.matrix[indices])[1, 0]

    def _run_decoding(self, mode):
        subjects = sorted(self._loader.datadir.joinpath("neural_data").glob("*.mat"))
        scores = np.zeros(len(subjects))
        for idx, subject in enumerate(subjects):
            X, Y, _ = self._loader.get_data_mvpa(subject)
            scores[idx] = self._calc_rsa(RDM(self._feature, X), RDM(self._target, Y))
        if mode == "score" and cache_subject_scores:
            temp_mode = "rsa_subjects"
            self._set_and_save(temp_mode, scores, self._get_fname(temp_mode))
        return scores.mean()

    def run(self, perms=True, iters=1000):
        self._run_pipeline("rsa_score")
        if perms:
            self._run_pipeline("rsa_null", iters)
            # self._plot()
        return self


class RDM:
    def __init__(self, name, samples):
        self._name = name
        self._samples = samples
        self._matrix = self._calc_matrix()

    @property
    def name(self):
        return self._name

    @property
    def matrix(self):
        return self._matrix

    def _calc_matrix(self):
        return 1 - np.corrcoef(self._samples)
