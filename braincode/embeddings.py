import typing
from abc import abstractmethod
from pathlib import Path

import numpy as np
from sklearn.random_projection import GaussianRandomProjection

from braincode.abstract import Object


class CodeModel(Object):
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

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
    def _embedding_models(self) -> typing.Dict[str, typing.Type[CodeModel]]:
        return {
            "code-projection": TokenProjection,
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
        raise NotImplementedError("under development")

    def _get_rep(self, program: str) -> np.ndarray:
        raise NotImplementedError("under development")
