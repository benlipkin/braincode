import typing

from braincode.analyses import BrainSimilarity
from braincode.metrics import *


class RSA(BrainSimilarity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def _similarity_metric(self):
        if self._metric:
            metric = globals()[self._metric]
            if not issubclass(metric, VectorMetric):
                raise ValueError("Invalid metric specified.")
            return RepresentationalSimilarity("correlation", metric())
        else:
            return RepresentationalSimilarity("correlation")


class CKA(BrainSimilarity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def _similarity_metric(self):
        return LinearCKA()
