from functools import partial

import braincode.metrics
from braincode.analyses import BrainSimilarity
from braincode.metrics import LinearCKA, RepresentationalSimilarity, VectorMetric


class RSA(BrainSimilarity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def _similarity_metric(self):
        rsa = partial(RepresentationalSimilarity, "correlation")
        if self._metric:
            metric = getattr(braincode.metrics, self._metric)
            if not issubclass(metric, VectorMetric):
                raise ValueError("Invalid metric specified.")
            return rsa(metric())
        return rsa()


class CKA(BrainSimilarity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def _similarity_metric(self):
        return LinearCKA()
