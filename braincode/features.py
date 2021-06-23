import builtins
import keyword
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import RobertaModel, RobertaTokenizer, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress tf warnings, e.g. cuda not found
logging.set_verbosity_error()  # suppress hf warnings, e.g. lmhead weights uninitialized


class FeatureExtractor:
    def __init__(self, feature):
        if feature == "code-bow":
            self._extractor = BagOfWords()
        elif feature == "code-tfidf":
            self._extractor = TFIDF()
        elif feature == "code-codeberta":
            self._extractor = CodeBERTa()
        else:
            raise ValueError("Feature not recognized. Select valid feature.")

    def fit_transform(self, programs):
        return StandardScaler().fit_transform(self._extractor.fit_transform(programs))


class CountVectorizer(ABC):
    def __init__(self):
        # filters = '!"#$&(),.:;?@[\\]^_`{|}~\t\n'  # leaving in <=>+-*/%
        self._tokenizer = Tokenizer()

    @property
    @abstractmethod
    def _mode(self):
        raise NotImplementedError("Handled by subclass.")

    @staticmethod
    def _clean_programs(programs):
        keywords = keyword.kwlist + dir(builtins)  # + dir(str) + dir(list) + dir(math)
        filters = "!#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n"  # leaving in _"
        subs = {}
        tokenizer = Tokenizer(filters=filters, lower=False)
        tokenizer.fit_on_texts(programs)
        for token in tokenizer.word_index.keys():
            if token in keywords:
                subs[token] = f" {token.upper()} "
            else:
                if '"' in token:
                    subs[token] = "   STR   "
                elif token.isdigit():
                    subs[token] = "   NUM   "
                else:
                    subs[token] = ""  # "   VAR   "
        for token in reversed(sorted(subs.keys(), key=len)):
            for idx in range(programs.shape[0]):
                programs[idx] = programs[idx].replace(token, subs[token])
        return programs

    def fit_transform(self, programs, clean_source_code=True):
        if clean_source_code:
            programs = self._clean_programs(programs)
        self._tokenizer.fit_on_texts(programs)
        return self._tokenizer.texts_to_matrix(programs, mode=self._mode)


class BagOfWords(CountVectorizer):
    @property
    def _mode(self):
        return "count"


class TFIDF(CountVectorizer):
    @property
    def _mode(self):
        return "tfidf"


class CodeBERTa:
    def __init__(self):
        self._spec = "huggingface/CodeBERTa-small-v1"
        self._cache_dir = Path(__file__).parent.joinpath(
            "outputs", "cache", "models", self._spec
        )
        if not self._cache_dir.exists():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = RobertaTokenizer.from_pretrained(
            self._spec, cache_dir=self._cache_dir
        )
        self._model = RobertaModel.from_pretrained(
            self._spec, cache_dir=self._cache_dir
        )

    def fit_transform(self, programs):
        outputs = []
        for program in programs:
            outputs.append(
                self._model(self._tokenizer.encode(program, return_tensors="pt"))
                .last_hidden_state.detach()
                .numpy()
                .mean(axis=1)
                .flatten()
            )
        return np.array(outputs)
        # only using last hidden state for now; can expand later if we want
