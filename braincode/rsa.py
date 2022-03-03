import numpy as np
from analyses import BrainAnalysis
from metrics import RepresentationalSimilarity


class RSA(BrainAnalysis):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, **kwargs)

    def _load_subject(self, subject):
        X, Y, _ = self._loader.get_data(self.__class__.__name__.lower(), subject)
        return X, Y, _

    def _shuffle(self, Y, _):
        np.random.shuffle(Y)
        return Y

    def _score(self, X, Y, _):
        metric = RepresentationalSimilarity("correlation")
        score = metric(X, Y)
        return score
