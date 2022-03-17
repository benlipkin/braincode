import numpy as np
from braincode.analyses import BrainMapping


class VWEA(BrainMapping):
    def __init__(self, feature: str, target: str, kwargs: dict) -> None:
        super().__init__(feature, target, **kwargs)


class NLEA(BrainMapping):
    def __init__(self, feature: str, target: str, kwargs: dict) -> None:
        super().__init__(feature, target, **kwargs)
