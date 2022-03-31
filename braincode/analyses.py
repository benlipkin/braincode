import logging
import os
import typing
from abc import ABC, abstractmethod
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

from braincode.data import *
from braincode.metrics import *
from braincode.plots import Plotter


class Analysis(ABC):
    def __init__(
        self,
        feature: str,
        target: str,
        base_path: Path = Path("braincode"),
        score_only: bool = True,
        debug: bool = False,
        code_model_dim: str = "",
    ) -> None:
        self._feature = feature
        self._target = target
        self._base_path = base_path
        self._score_only = score_only
        self._debug = debug
        self._code_model_dim = code_model_dim
        if "code-" not in self.target:
            self._code_model_dim = ""
        self._name = self.__class__.__name__
        self._loader = globals()[f"DataLoader{self._name}"](
            self._base_path, self.feature, self.target
        )
        self._logger = logging.getLogger(self._name)
        self._score = None
        self._null = None

    @property
    def feature(self) -> str:
        return self._feature

    @property
    def target(self) -> str:
        return self._target

    @property
    def score(self) -> np.float:
        if not self._score:
            raise RuntimeError("Score not set. Need to run.")
        return self._score

    @property
    def null(self) -> np.ndarray:
        if not self._null:
            raise RuntimeError("Null not set. Need to run.")
        return self._null

    @property
    def pval(self) -> np.float:
        return (self.score < self.null).sum() / self.null.size

    def _get_fname(self, mode: str) -> Path:
        return Path(
            os.path.join(
                self._base_path,
                ".cache",
                "scores",
                self._name.lower(),
                f"{mode}_{self.feature.split('-')[1]}_{self.target.split('-')[1]}{self._code_model_dim}.npy",
            )
        )

    def _set_and_save(
        self, mode: str, val: typing.Union[np.float, np.ndarray], fname: Path
    ) -> None:
        setattr(self, f"_{mode}", val)
        if not self._debug:
            np.save(fname, val)
        tag = f": {val:.3f}" if mode == "score" else ""
        self._logger.info(f"Caching '{fname.name}'{tag}.")

    def _run_pipeline(self, mode: str, iters: int = 1) -> None:
        if mode not in ["score", "null"]:
            raise RuntimeError("Mode set incorrectly. Must be 'score' or 'null'")
        fname = self._get_fname(mode)
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
        if fname.exists() and not self._debug:
            setattr(self, f"_{mode}", np.load(fname, allow_pickle=True))
            self._logger.info(f"Loading '{fname.name}' from cache.")
            return
        samples = np.zeros((iters))
        for idx in tqdm(range(iters)):
            score = self._run_mapping(mode)
            if mode == "score":
                self._set_and_save(mode, score, fname)
                return
            samples[idx] = score
        self._set_and_save(mode, samples, fname)

    @abstractmethod
    def _run_mapping(self, mode: str) -> typing.Union[np.float, np.ndarray]:
        raise NotImplementedError("Handled by subclass.")

    def _plot(self) -> None:
        Plotter(self).plot()

    def run(self, iters: int = 1000, plot: bool = False) -> None:
        self._run_pipeline("score")
        if not self._score_only:
            self._run_pipeline("null", iters)
            if plot:
                self._plot()


class BrainAnalysis(Analysis):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def subjects(self) -> typing.List[Path]:
        return [
            s
            for s in sorted(self._loader.datadir.joinpath("neural_data").glob("*.mat"))
            if "737" not in str(s)  # remove this subject as in Ivanova et al (2020)
        ]

    @abstractmethod
    def _load_subject(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Handled by subclass.")

    @abstractmethod
    def _shuffle(self, Y: np.ndarray, runs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Handled by subclass.")

    @abstractmethod
    def _calc_score(self, X: np.ndarray, Y: np.ndarray, runs: np.ndarray) -> np.float:
        raise NotImplementedError("Handled by subclass.")

    def _run_mapping(self, mode: str, cache_subject_scores: bool = True) -> np.float:
        scores = np.zeros(len(self.subjects))
        for idx, subject in enumerate(self.subjects):
            X, Y, runs = self._load_subject(subject)
            if mode == "null":
                Y = self._shuffle(Y, runs)
            scores[idx] = self._calc_score(X, Y, runs)
        if mode == "score" and cache_subject_scores:
            temp_mode = "subjects"
            self._set_and_save(temp_mode, scores, self._get_fname(temp_mode))
        return scores.mean()


class Mapping(Analysis):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_metric(Y: np.ndarray) -> Metric:
        if Y.ndim == 1:
            return ClassificationAccuracy()
        elif Y.ndim == 2:
            if Y.shape[1] == 1:
                return PearsonR()
            else:
                return RankAccuracy()
        else:
            raise NotImplementedError("Metrics only defined for 1D and 2D arrays.")

    def _cross_validate_model(
        self, X: np.ndarray, Y: np.ndarray, runs: np.ndarray
    ) -> np.float:
        if any(a.shape[0] != b.shape[0] for a, b in combinations([X, Y, runs], 2)):
            raise ValueError("X Y and runs must all have the same number of samples.")
        model_class = RidgeClassifierCV if Y.ndim == 1 else RidgeCV
        scores = np.zeros(np.unique(runs).size)
        for idx, (train, test) in enumerate(LeaveOneGroupOut().split(X, Y, runs)):
            model = model_class(alphas=np.logspace(-2, 2, 9)).fit(X[train], Y[train])
            metric = self._get_metric(Y)
            scores[idx] = metric(model.predict(X[test]), Y[test])
        return scores.mean()


class BrainMapping(BrainAnalysis, Mapping):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _load_subject(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Y, runs = self._loader.get_data(
            self._name.lower(), subject, self._debug, self._code_model_dim
        )
        return X, Y, runs

    @staticmethod
    def _shuffle(Y_in: np.ndarray, runs: np.ndarray) -> np.ndarray:
        if Y_in.shape[0] != runs.shape[0]:
            raise ValueError("Y and runs must have the same number of samples.")
        Y_out = np.zeros(Y_in.shape)
        for run in np.unique(runs):
            Y_out[runs == run] = np.random.permutation(Y_in[runs == run])
        return Y_out

    def _calc_score(self, X: np.ndarray, Y: np.ndarray, runs: np.ndarray) -> np.float:
        score = self._cross_validate_model(X, Y, runs)
        return score


class BrainSimilarity(BrainAnalysis):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _load_subject(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, typing.Any]:
        X, Y, _ = self._loader.get_data(self._name.lower(), subject, self._debug)
        return X, Y, _

    @staticmethod
    def _shuffle(Y: np.ndarray, _: np.ndarray) -> np.ndarray:
        np.random.shuffle(Y)
        return Y

    def _calc_score(self, X: np.ndarray, Y: np.ndarray, _: np.ndarray) -> np.float:
        score = self._similarity_metric(X, Y)
        return score

    @property
    @abstractmethod
    def _similarity_metric(self) -> MatrixMetric:
        raise NotImplementedError("Handled by subclass.")
