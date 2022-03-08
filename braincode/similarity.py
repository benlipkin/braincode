from analyses import BrainSimilarity
from metrics import LinearCKA, RepresentationalSimilarity


class RSA(BrainSimilarity):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, **kwargs)

    @property
    def _similarity_metric(self):
        return RepresentationalSimilarity("correlation")


class CKA(BrainSimilarity):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, **kwargs)

    @property
    def _similarity_metric(self):
        return LinearCKA()
