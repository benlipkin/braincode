import multiprocessing
import os
import pickle as pkl
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from code_seq2seq.representations import get_representation
from code_seq2seq.tokenize import _tokenize_programs as tokenize_programs
from code_seq2seq.train import params
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
from sklearn.random_projection import GaussianRandomProjection
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import AutoModel, AutoTokenizer

# import openai

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"


class ProgramEmbedder:
    def __init__(self, embedder, base_path, code_model_dim):
        if embedder == "code-projection":
            self._embedder = TokenProjection(base_path)
        elif embedder == "code-bow":
            self._embedder = BagOfWords(base_path)
        elif embedder == "code-tfidf":
            self._embedder = TFIDF(base_path)
        elif embedder == "code-seq2seq":
            self._embedder = CodeSeq2Seq(base_path)
        elif embedder == "code-transformer":
            self._embedder = CodeTransformer(base_path)
        elif embedder == "code-xlnet":
            self._embedder = CodeXLNet(base_path)
        elif embedder == "code-bert":
            self._embedder = CodeBERT(base_path)
        elif embedder == "code-gpt2":
            self._embedder = CodeGPT2(base_path)
        elif embedder == "code-roberta":
            self._embedder = CodeBERTa(base_path)
        elif embedder == "code-ada":
            self._embedder = AdaGPT3(base_path)
        elif embedder == "code-babbage":
            self._embedder = BabbageGPT3(base_path)
        else:
            raise ValueError("Embedder not recognized. Select valid embedding model.")
        self._code_model_dim = code_model_dim

    def fit_transform(self, programs):
        if not callable(getattr(self._embedder, "fit_transform", None)):
            raise NotImplementedError(
                f"{self._embedder.__class__.__name__} must implement 'fit_transform' method."
            )
        embedding = self._embedder.fit_transform(programs)
        if self._code_model_dim != "":
            embedding = GaussianRandomProjection(
                n_components=int(self._code_model_dim), random_state=0
            ).fit_transform(embedding)
        return embedding


class TokenProjection:
    def __init__(self, base_path):
        self._base_path = base_path
        seq2seq_cfg = CodeSeq2Seq(base_path)
        self._vocab = seq2seq_cfg._vocab
        self._vocab_size = len(self._vocab)
        self._embedding_size = seq2seq_cfg._model.encoder.hidden_size
        self._random_matrix = np.random.default_rng(0).standard_normal(
            (self._vocab_size, self._embedding_size)
        )

    def _get_rep(self, program):
        rep = np.zeros(self._embedding_size)
        for token in (tokenize_programs([program])[0]).split():
            rep += self._random_matrix[self._vocab[token], :]
        return rep

    def fit_transform(self, programs):
        outputs = []
        for program in programs:
            outputs.append(self._get_rep(program))
        return np.array(outputs)


class CountVectorizer(ABC):
    def __init__(self, base_path):
        cache_dir = Path(os.path.join(base_path, ".cache", "datasets", "huggingface"))
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._dataset = load_dataset(
            "code_search_net", "python", split="validation", cache_dir=cache_dir
        )["func_code_string"]
        self._model = Tokenizer(num_words=TokenProjection(base_path)._vocab_size)

    @property
    @abstractmethod
    def _mode(self):
        raise NotImplementedError("Handled by subclass.")

    def fit_transform(self, programs):
        self._model.fit_on_texts(tokenize_programs(self._dataset))
        outputs = self._model.texts_to_matrix(
            tokenize_programs(programs), mode=self._mode
        )
        return outputs[:, np.any(outputs, axis=0)]


class BagOfWords(CountVectorizer):
    def __init__(self, base_path):
        super().__init__(base_path)

    @property
    def _mode(self):
        return "count"


class TFIDF(CountVectorizer):
    def __init__(self, base_path):
        super().__init__(base_path)

    @property
    def _mode(self):
        return "tfidf"


class DNN(ABC):
    def __init__(self, base_path):
        self._base_path = base_path

    @staticmethod
    def _get_rep(forward_output):
        rep = forward_output[0].mean(axis=1)
        if rep.device != "cpu":
            rep = rep.cpu()
        return rep.detach().numpy().squeeze()

    @abstractmethod
    def _forward_pipeline(program):
        raise NotImplementedError()

    def fit_transform(self, programs):
        outputs = []
        for program in programs:
            outputs.append(self._get_rep(self._forward_pipeline(program)))
        return np.array(outputs)


class CodeSeq2Seq(DNN):
    def __init__(self, base_path):
        super().__init__(base_path)
        cache_dir = Path(
            os.path.join(self._base_path, ".cache", "models", "code_seq2seq")
        )
        if torch.cuda.is_available():
            device_count = torch.torch.cuda.device_count()
            if device_count > 0:
                device_id = random.randrange(device_count)
                self._device = torch.device("cuda:" + str(device_id))
                torch.cuda.set_device(self._device)
        else:
            self._device = "cpu"
        with open(cache_dir.joinpath("code_seq2seq_py8kcodenet.torch"), "rb") as fp:
            self._model = torch.load(fp, map_location=self._device)
        with open(cache_dir.joinpath("vocab_code_seq2seq_py8kcodenet.pkl"), "rb") as fp:
            self._vocab = pkl.load(fp)
        self._max_seq_len = params["max_len"]

    @staticmethod
    def _get_rep(rep):
        if rep.device != "cpu":
            rep = rep.cpu()
        return rep.detach().numpy().squeeze()

    def _forward_pipeline(self, program):
        return get_representation(
            self._model,
            tokenize_programs([program])[0],
            self._max_seq_len,
            self._vocab,
            self._device,
        )


class ZuegnerModel(DNN):
    def __init__(self, base_path):
        super().__init__(base_path)
        model_manager = get_model_manager(self._model_type)
        self._model_config = model_manager.load_config(self._run_id)
        data_manager = CTPreprocessedDataManager(
            DATA_PATH_STAGE_2, self._model_config["data_setup"]["language"]
        )
        self._data_config = data_manager.load_config()
        self._model = model_manager.load_model(self._run_id, "latest", gpu=False).eval()
        self._distances_transformer = self._build_distances_transformer()
        self._vocabulary_transformer = VocabularyTransformer(
            *data_manager.load_vocabularies()
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

    def _forward_pipeline(self, program):
        stage1_sample = CTStage1Preprocessor("python").process(
            [("f", "", self._prep_program(program))],
            multiprocessing.current_process().pid,
        )
        stage2_sample = stage1_sample[0]
        if self._data_config["preprocessing"]["remove_punctuation"]:
            stage2_sample.remove_punctuation()
        batch = make_batch_from_sample(
            self._distances_transformer(self._vocabulary_transformer(stage2_sample)),
            self._model_config,
            self._model_type,
        )
        return self._forward(batch)


class CodeXLNet(ZuegnerModel):
    def __init__(self, base_path):
        super().__init__(base_path)

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
    def __init__(self, base_path):
        super().__init__(base_path)

    @property
    def _model_type(self):
        return "code_transformer"

    @property
    def _run_id(self):
        return "CT-5"

    def _forward(self, batch):
        return self._model.lm_encoder.forward_batch(batch, need_all_embeddings=True)


class HFModel(DNN):
    def __init__(self, base_path):
        super().__init__(base_path)
        cache_dir = Path(
            os.path.join(
                self._base_path,
                ".cache",
                "models",
                self._spec.split(os.sep)[0],
                self._spec.split(os.sep)[-1],
            )
        )
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self._spec, cache_dir=cache_dir)
        self._model = AutoModel.from_pretrained(self._spec, cache_dir=cache_dir)

    @property
    @abstractmethod
    def _spec(self):
        raise NotImplementedError()

    def _forward_pipeline(self, program):
        return self._model.forward(self._tokenizer.encode(program, return_tensors="pt"))


class CodeBERT(HFModel):
    def __init__(self, base_path):
        super().__init__(base_path)

    @property
    def _spec(self):
        return "microsoft/codebert-base-mlm"


class CodeGPT2(HFModel):
    def __init__(self, base_path):
        super().__init__(base_path)

    @property
    def _spec(self):
        return "microsoft/CodeGPT-small-py"


class CodeBERTa(HFModel):
    def __init__(self, base_path):
        super().__init__(base_path)

    @property
    def _spec(self):
        return "huggingface/CodeBERTa-small-v1"


class OpenAiGPT3:
    def __init__(self, base_path):
        self._base_path = base_path
        openai.api_key_path = Path(os.path.join(self._base_path, ".openai_api_key"))

    @property
    @abstractmethod
    def _engine(self):
        raise NotImplementedError()

    def _get_rep(self, program):
        return openai.Engine(id=self._engine).embeddings(
            input=[program.replace("\n", " ")]
        )["data"][0]["embedding"]

    def fit_transform(self, programs):
        outputs = []
        for program in programs:
            outputs.append(self._get_rep(program))
        return np.array(outputs)


class AdaGPT3(OpenAiGPT3):
    def __init__(self, base_path):
        super().__init__(base_path)

    @property
    def _engine(self):
        return "ada-code-search-code"


class BabbageGPT3(OpenAiGPT3):
    def __init__(self, base_path):
        super().__init__(base_path)

    @property
    def _engine(self):
        return "babbage-code-search-code"
