import numpy as np


class FeatureExtractor:
    def __init__(self, feature):
        if feature == "bow":
            self._extractor = BagOfWords()
        else:
            raise NotImplementedError()

    def fit_transform(self, programs):
        return self._extractor.fit_transform(programs)


class BagOfWords:
    def __init__(self):
        pass

    def fit_transform(self, programs):
        print(programs)  # note to self: resume here
        raise NotImplementedError()
