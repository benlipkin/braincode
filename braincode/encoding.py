import numpy as np
from decoding import BrainMapping


class NLEA(BrainMapping):  # network-level encoding analysis
    def __init__(self, feature, target, kwargs):
        super().__init__(feature, target, kwargs)
