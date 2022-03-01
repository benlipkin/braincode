import numpy as np
from decoding import BrainMapping


class NLEA(BrainMapping):
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, kwargs)
