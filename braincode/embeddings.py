import typing
from abc import abstractmethod
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.random_projection import GaussianRandomProjection

from braincode.abstract import Object


class CodeModel(Object):
    def __init__(self, base_path: Path) -> None:
        super().__init__(base_path)

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
            "code-graph": GraphProjection,
            "code-llm_350m_nl": partial(HFModel, "Salesforce/codegen-350M-nl"),
            "code-llm_350m_mono": partial(HFModel, "Salesforce/codegen-350M-mono"),
            "code-llm_350m_multi": partial(HFModel, "Salesforce/codegen-350M-multi"),
            "code-llm_2b_nl": partial(HFModel, "Salesforce/codegen-2B-nl"),
            "code-llm_2b_mono": partial(HFModel, "Salesforce/codegen-2B-mono"),
            "code-llm_2b_multi": partial(HFModel, "Salesforce/codegen-2B-multi"),
            "code-llm_6b_nl": partial(HFModel, "Salesforce/codegen-6B-nl"),
            "code-llm_6b_mono": partial(HFModel, "Salesforce/codegen-6B-mono"),
            "code-llm_6b_multi": partial(HFModel, "Salesforce/codegen-6B-multi"),
            "code-llm_16b_nl": partial(HFModel, "Salesforce/codegen-16B-nl"),
            "code-llm_16b_mono": partial(HFModel, "Salesforce/codegen-16B-mono"),
            "code-llm_16b_multi": partial(HFModel, "Salesforce/codegen-16B-multi"),
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
        # Use CodeGen tokenizer to get vocab size and tokenize program
        raise NotImplementedError("under development")

    def _get_rep(self, program: str) -> np.ndarray:
        raise NotImplementedError("under development")


class GraphProjection(CodeModel):
    def __init__(self, base_path: Path) -> None:
        super().__init__(base_path)
        raise NotImplementedError("under development")

    def _get_rep(self, program: str) -> np.ndarray:
        raise NotImplementedError("under development")


class HFModel(CodeModel):
    def __init__(self, config: str, base_path: Path) -> None:
        super().__init__(base_path)
        raise NotImplementedError("under development")

    def _get_rep(self, program: str) -> np.ndarray:
        raise NotImplementedError("under development")
