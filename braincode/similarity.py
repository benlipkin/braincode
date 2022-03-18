import typing

from braincode.analyses import BrainSimilarity
from braincode.metrics import LinearCKA, RepresentationalSimilarity


class RSA(BrainSimilarity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def _similarity_metric(self):
        return RepresentationalSimilarity("correlation")


class CKA(BrainSimilarity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def _similarity_metric(self):
        return LinearCKA()
