import builtins
import keyword
import math
import os
import re
from abc import ABC, abstractmethod

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress tf warnings, e.g. cuda not found

from tensorflow.keras.preprocessing.text import Tokenizer


class FeatureExtractor:
    def __init__(self, feature):
        if feature == "bow":
            self._extractor = BagOfWords()
        elif feature == "tfidf":
            self._extractor = TFIDF()
        else:
            raise NotImplementedError()

    def fit_transform(self, programs):
        return self._extractor.fit_transform(programs)


class CountVectorizer(ABC):
    def __init__(self):
        self.tokenizer = None

    @property
    @abstractmethod
    def _mode(self):
        raise NotImplementedError()

    @staticmethod
    def _clean_programs(programs):
        keywords = keyword.kwlist + dir(builtins) + dir(str) + dir(list) + dir(math)
        filters = "!#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n"  # leaving in _"
        subs = {}
        tokenizer = Tokenizer(filters=filters, lower=False)
        tokenizer.fit_on_texts(programs)
        for token in tokenizer.word_index.keys():
            if token in keywords:
                subs[token] = f" {token.upper()} "
            else:
                if '"' in token:
                    subs[token] = " STR "
                elif token.isdigit():
                    subs[token] = " NUM "
                else:
                    subs[token] = " VAR "
        for token in reversed(sorted(subs.keys(), key=len)):
            for idx in range(programs.shape[0]):
                programs[idx] = programs[idx].replace(token, subs[token])
        return programs

    def _fit(self, programs):
        filters = '!"#$&(),.:;?@[\\]^_`{|}~\t\n'  # leaving in <=>+-*/%
        self.tokenizer = Tokenizer(filters=filters)
        self.tokenizer.fit_on_texts(programs)

    def _transform(self, programs):
        return self.tokenizer.texts_to_matrix(programs, mode=self._mode)

    def fit_transform(self, programs, clean_source_code=True):
        if clean_source_code:
            programs = self._clean_programs(programs)
        self._fit(programs)
        return self._transform(programs)


class BagOfWords(CountVectorizer):
    @property
    def _mode(self):
        return "count"


class TFIDF(CountVectorizer):
    @property
    def _mode(self):
        return "tfidf"
