import numpy as np
from analyses import BrainMapping, Mapping


class MVPA(BrainMapping):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, **kwargs)


class PRDA(Mapping):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, **kwargs)

    def _run_mapping(self, mode):
        X, Y, runs = self._loader.get_data(self.__class__.__name__.lower())
        if mode == "null":
            np.random.shuffle(Y)
        return self._cross_validate_model(X, Y, runs)
