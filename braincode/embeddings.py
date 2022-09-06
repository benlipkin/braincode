import typing
import warnings
from abc import abstractmethod
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.random_projection import GaussianRandomProjection
from transformers import CodeGenConfig, CodeGenTokenizer, CodeGenModel, logging

from braincode.abstract import Object

logging.set_verbosity_error()


class CodeModel(Object):
    def __init__(self, base_path: Path) -> None:
        super().__init__(base_path)
        self._cache_dir = base_path.joinpath(".cache", "models", self._name)

    @abstractmethod
    def _get_rep(self, program: str) -> np.ndarray:
        raise NotImplementedError()

    def fit_transform(self, programs: np.ndarray) -> np.ndarray:
        outputs = []
        for program in programs:
            outputs.append(self._get_rep(program))
        return np.array(outputs)


class ProgramEmbedder:
    def __init__(self, embedder: str, base_path: Path, code_model_dim: str) -> None:
        self._embedder = self._embedding_models[embedder](base_path)
        self._code_model_dim = code_model_dim

    @property
    def _embedding_models(
        self,
    ) -> typing.Dict[str, typing.Union[typing.Type[CodeModel], partial[CodeModel]]]:
        return {
            "code-tokens": TokenProjection,
            "code-llm_350m_nl": partial(HFCodeGen, "Salesforce/codegen-350M-nl"),
            "code-llm_2b_nl": partial(HFCodeGen, "Salesforce/codegen-2B-nl"),
            "code-llm_6b_nl": partial(HFCodeGen, "Salesforce/codegen-6B-nl"),
            "code-llm_16b_nl": partial(HFCodeGen, "Salesforce/codegen-16B-nl"),
            "code-llm_350m_mono": partial(HFCodeGen, "Salesforce/codegen-350M-mono"),
            "code-llm_2b_mono": partial(HFCodeGen, "Salesforce/codegen-2B-mono"),
            "code-llm_6b_mono": partial(HFCodeGen, "Salesforce/codegen-6B-mono"),
            "code-llm_16b_mono": partial(HFCodeGen, "Salesforce/codegen-16B-mono"),
        }

    def fit_transform(self, programs: np.ndarray) -> np.ndarray:
        embedding = self._embedder.fit_transform(programs)
        if self._code_model_dim != "":
            embedding = GaussianRandomProjection(
                n_components=int(self._code_model_dim), random_state=0
            ).fit_transform(embedding)
        return embedding


class TokenProjection(CodeModel):
    def __init__(self, base_path: Path) -> None:
        super().__init__(base_path)
        default_id = "Salesforce/codegen-350M-mono"
        cfg = CodeGenConfig.from_pretrained(default_id, cache_dir=self._cache_dir)
        self._tokenizer = CodeGenTokenizer.from_pretrained(
            default_id, cache_dir=self._cache_dir
        )
        self._projection = np.random.default_rng(0).standard_normal(
            (cfg.vocab_size, cfg.n_embd)
        )

    def _get_rep(self, program: str) -> np.ndarray:
        rep = np.zeros(self._projection.shape[1])
        for token in self._tokenizer(program)["input_ids"]:
            rep += self._projection[token, :]
        return rep


class HFCodeGen(CodeModel):
    def __init__(self, spec: str, base_path: Path) -> None:
        super().__init__(base_path)
        self._tokenizer = CodeGenTokenizer.from_pretrained(
            spec, cache_dir=self._cache_dir
        )
        self._model = CodeGenModel.from_pretrained(
            spec, cache_dir=self._cache_dir, torch_dtype=torch.bfloat16
        )
        self._set_torch_device()

    def _set_torch_device(self) -> None:
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # type: ignore
            try:
                self._model = self._model.to(self._device)
                self._logger.info("Model inference running on GPU.")
                return
            except RuntimeError as e:
                self._logger.error(e)
                self._device = torch.device("cpu")
                torch.set_default_tensor_type(torch.BFloat16Tensor)  # type: ignore
                self._model = self._model.to(self._device)
                self._logger.warning("GPU error. Model inference running on CPU.")
        else:
            self._device = torch.device("cpu")
            torch.set_default_tensor_type(torch.BFloat16Tensor)  # type: ignore
            self._model = self._model.to(self._device)
            self._logger.info("No GPU found. Model inference running on CPU.")

    def _get_rep(self, program: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self._tokenizer(program, return_tensors="pt").to(self._device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = self._model(**inputs)
            embedding = (
                outputs.last_hidden_state.mean(axis=1)
                .cpu()
                .detach()
                .float()
                .numpy()
                .squeeze()
            )
        return embedding
