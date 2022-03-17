from braincode.analyses import BrainSimilarity
from braincode.metrics import LinearCKA, RepresentationalSimilarity


class RSA(BrainSimilarity):
    def __init__(self, feature: str, target: str, kwargs: dict) -> None:
        super().__init__(feature, target, **kwargs)

    @property
    def _similarity_metric(self):
        return RepresentationalSimilarity("correlation")


class CKA(BrainSimilarity):
    def __init__(self, feature: str, target: str, kwargs: dict) -> None:
        super().__init__(feature, target, **kwargs)

    @property
    def _similarity_metric(self):
        return LinearCKA()
