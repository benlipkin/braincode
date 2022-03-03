import numpy as np
from decoding import BrainMapping


class VWEA(BrainMapping):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, **kwargs)


class NLEA(BrainMapping):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, **kwargs)
