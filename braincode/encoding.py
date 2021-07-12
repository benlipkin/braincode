import builtins
import io
import keyword
import os
import token
from abc import ABC, abstractmethod
from pathlib import Path
from tokenize import tokenize

import numpy as np
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import RobertaModel, RobertaTokenizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"


class ProgramEncoder:
    def __init__(self, encoder):
        if encoder == "code-bow":
            self._encoder = BagOfWords()
        elif encoder == "code-tfidf":
            self._encoder = TFIDF()
        elif encoder == "code-codeberta":
            self._encoder = CodeBERTa()
        else:
            raise ValueError("Encoder not recognized. Select valid encoder.")

    def fit_transform(self, programs):
        if not callable(getattr(self._encoder, "fit_transform", None)):
            raise NotImplementedError(
                f"{self._encoder.__class__.__name__} must implement 'fit_transform' method."
            )
        return self._encoder.fit_transform(programs)


class CountVectorizer(ABC):
    def __init__(self):
        self._cache_dir = Path(__file__).parent.joinpath("outputs", "cache", "datasets")
        if not self._cache_dir.exists():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._dataset = load_dataset(
            "code_search_net", "python", split="validation", cache_dir=self._cache_dir
        )["func_code_string"]
        self._model = Tokenizer(num_words=200)  # arbitrary N > vocab size

    @property
    @abstractmethod
    def _mode(self):
        raise NotImplementedError("Handled by subclass.")

    @staticmethod
    def _tokenize_programs(programs):
        sequences = []
        tokens = keyword.kwlist + dir(builtins)
        for program in programs:
            sequence = []
            for type, text, _, _, _ in tokenize(io.BytesIO(program.encode()).readline):
                if type is token.STRING:
                    sequence.append(1)
                elif type is token.NUMBER:
                    sequence.append(2)
                elif text in tokens:
                    sequence.append(3 + tokens.index(text))
                else:
                    continue
            sequences.append(sequence)
        return sequences

    def fit_transform(self, programs):
        self._model.fit_on_sequences(self._tokenize_programs(self._dataset))
        outputs = self._model.sequences_to_matrix(
            self._tokenize_programs(programs), mode=self._mode
        )
        return outputs[:, np.any(outputs, axis=0)]


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
