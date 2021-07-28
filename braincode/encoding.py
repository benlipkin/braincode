import builtins
import io
import keyword
import os
import token
from abc import ABC, abstractmethod
from pathlib import Path
from tokenize import tokenize

import numpy as np
from code_transformer.env import DATA_PATH_STAGE_2
from code_transformer.preprocessing.datamanager.preprocessed import \
    CTPreprocessedDataManager
from code_transformer.preprocessing.graph.binning import ExponentialBinning
from code_transformer.preprocessing.graph.distances import (
    AncestorShortestPaths, DistanceBinning, PersonalizedPageRank,
    ShortestPaths, SiblingShortestPaths)
from code_transformer.preprocessing.graph.transform import DistancesTransformer
from code_transformer.preprocessing.nlp.vocab import VocabularyTransformer
from code_transformer.preprocessing.pipeline.stage1 import CTStage1Preprocessor
from code_transformer.utils.inference import (get_model_manager,
                                              make_batch_from_sample)
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
        elif encoder == "code-xlnet":
            self._encoder = XLNet()
        elif encoder == "code-ct":
            self._encoder = CodeTransformer()
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
        cache_dir = Path(__file__).parent.joinpath(".cache", "datasets", "huggingface")
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._dataset = load_dataset(
            "code_search_net", "python", split="validation", cache_dir=cache_dir
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


class ZuegnerModel(ABC):
    def __init__(self):
        self._language = "python"
        self._model_manager = get_model_manager(self._model_type)
        self._model_config = self._model_manager.load_config(self._run_id)
        self._model = self._model_manager.load_model(
            self._run_id, "latest", gpu=False
        ).eval()
        self._data_manager = CTPreprocessedDataManager(
            DATA_PATH_STAGE_2,
            self._model_config["data_setup"]["language"],
            partition="train",
            shuffle=True,
        )
        self._data_config = self._data_manager.load_config()
        self._distances_transformer = self._build_distances_transformer()
        self._vocabulary_transformer = VocabularyTransformer(
            *self._data_manager.load_vocabularies()
        )

    @property
    @abstractmethod
    def _model_type(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def _run_id(self):
        raise NotImplementedError()

    def _build_distances_transformer(self):
        distances_config = self._data_config["distances"]
        binning_config = self._data_config["binning"]
        return DistancesTransformer(
            [
                PersonalizedPageRank(
                    threshold=distances_config["ppr_threshold"],
                    log=distances_config["ppr_use_log"],
                    alpha=distances_config["ppr_alpha"],
                ),
                ShortestPaths(threshold=distances_config["sp_threshold"]),
                AncestorShortestPaths(
                    forward=distances_config["ancestor_sp_forward"],
                    backward=distances_config["ancestor_sp_backward"],
                    negative_reverse_dists=distances_config[
                        "ancestor_sp_negative_reverse_dists"
                    ],
                    threshold=distances_config["ancestor_sp_threshold"],
                ),
                SiblingShortestPaths(
                    forward=distances_config["sibling_sp_forward"],
                    backward=distances_config["sibling_sp_backward"],
                    negative_reverse_dists=distances_config[
                        "sibling_sp_negative_reverse_dists"
                    ],
                    threshold=distances_config["sibling_sp_threshold"],
                ),
            ],
            DistanceBinning(
                binning_config["num_bins"],
                binning_config["n_fixed_bins"],
                ExponentialBinning(binning_config["exponential_binning_growth_factor"]),
            ),
        )

    @staticmethod
    def _prep_program(program):
        return f"def f():\n{program}".replace("\n", "\n    ")

    @abstractmethod
    def _forward(self, batch):
        raise NotImplementedError()

    def fit_transform(self, programs):
        outputs = []
        for program in programs:
            stage1_sample = CTStage1Preprocessor(self._language).process(
                [("f", "", self._prep_program(program))], 0
            )
            stage2_sample = stage1_sample[0]
            if self._data_config["preprocessing"]["remove_punctuation"]:
                stage2_sample.remove_punctuation()
            batch = make_batch_from_sample(
                self._distances_transformer(
                    self._vocabulary_transformer(stage2_sample)
                ),
                self._model_config,
                self._model_type,
            )
            outputs.append(
                self._forward(batch).all_emb[-1][1].detach().numpy().flatten()
            )
        return np.array(outputs)


class XLNet(ZuegnerModel):
    @property
    def _model_type(self):
        return "xl_net"

    @property
    def _run_id(self):
        return "XL-1"

    def _forward(self, batch):
        return self._model.lm_encoder.forward(
            input_ids=batch.tokens,
            pad_mask=batch.pad_mask,
            attention_mask=batch.perm_mask,
            target_mapping=batch.target_mapping,
            token_type_ids=batch.token_types,
            languages=batch.languages,
            need_all_embeddings=True,
        )


class CodeTransformer(ZuegnerModel):
    @property
    def _model_type(self):
        return "code_transformer"

    @property
    def _run_id(self):
        return "CT-5"

    def _forward(self, batch):
        return self._model.lm_encoder.forward_batch(batch, need_all_embeddings=True)


class CodeBERTa:
    def __init__(self):
        spec = "huggingface/CodeBERTa-small-v1"
        cache_dir = Path(__file__).parent.joinpath(
            ".cache", "models", "huggingface", spec.split("/")[-1]
        )
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = RobertaTokenizer.from_pretrained(spec, cache_dir=cache_dir)
        self._model = RobertaModel.from_pretrained(spec, cache_dir=cache_dir)

    def fit_transform(self, programs):
        outputs = []
        for program in programs:
            outputs.append(
                self._model.forward(
                    self._tokenizer.encode(program, return_tensors="pt")
                )[1]
                .detach()
                .numpy()
                .flatten()
            )
        return np.array(outputs)
