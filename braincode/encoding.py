import numpy as np
from analyses import BrainMapping


class VWEA(BrainMapping):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, **kwargs)


class NLEA(BrainMapping):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, **kwargs)
