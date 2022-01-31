import numpy as np
from decoding import Analysis
from scipy.stats import pearsonr


class RSA(Analysis):
    def __init__(self, feature, target, base_path, score_only, code_model_dim):
        super().__init__(feature, target, base_path, score_only, code_model_dim)

    @staticmethod
    def _calc_rsa(brain_rdm, model_rdm):
        if not brain_rdm.matrix.size == model_rdm.matrix.size:
            raise RuntimeError("RDMs mismatched. Check feature target pair.")
        indices = np.triu_indices(brain_rdm.matrix.shape[0], k=1)
        score, pval = pearsonr(brain_rdm.matrix[indices], model_rdm.matrix[indices])
        return score

    def _run_decoding(self, mode, cache_subject_scores=True):
        subjects = sorted(self._loader.datadir.joinpath("neural_data").glob("*.mat"))
        subjects = [
            s for s in subjects if "737" not in str(s)
        ]  # remove this subject as in Ivanova et al (2020)
        scores = np.zeros(len(subjects))
        for idx, subject in enumerate(subjects):
            X, Y, _ = self._loader.get_data(self.__class__.__name__.lower(), subject)
            if mode == "null":
                np.random.shuffle(Y)
            scores[idx] = self._calc_rsa(RDM(self._feature, X), RDM(self._target, Y))
        if mode == "score" and cache_subject_scores:
            temp_mode = "subjects"
            self._set_and_save(temp_mode, scores, self._get_fname(temp_mode))
        return scores.mean()


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
