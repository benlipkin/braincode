import numpy as np
from decoding import BrainAnalysis
from scipy.stats import pearsonr


class RSA(BrainAnalysis):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, **kwargs)

    @staticmethod
    def _calc_rsa(brain_rdm, model_rdm):
        if not brain_rdm.matrix.size == model_rdm.matrix.size:
            raise RuntimeError("RDMs mismatched. Check feature target pair.")
        indices = np.triu_indices(brain_rdm.matrix.shape[0], k=1)
        score, pval = pearsonr(brain_rdm.matrix[indices], model_rdm.matrix[indices])
        return score

    def _load_subject(self, subject):
        X, Y, _ = self._loader.get_data(self.__class__.__name__.lower(), subject)
        return X, Y, _

    def _shuffle(self, Y, _):
        np.random.shuffle(Y)
        return Y

    def _score(self, X, Y, _):
        score = self._calc_rsa(RDM(self._feature, X), RDM(self._target, Y))
        return score


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
