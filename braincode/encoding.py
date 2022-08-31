import numpy as np

from braincode.analyses import BrainMapping
from braincode.decoding import PRDA


class VWEA(BrainMapping):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class NLEA(BrainMapping):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class PREA(PRDA):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
