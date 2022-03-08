import logging
import os
from abc import ABC, abstractmethod
from itertools import combinations
from pathlib import Path

import numpy as np
from data import DataLoader
from metrics import ClassificationAccuracy, PearsonR, RankAccuracy
from plots import Plotter
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm


class Analysis(ABC):
    def __init__(self, feature, target, base_path, score_only, code_model_dim):
        self._feature = feature
        self._target = target
        self._base_path = base_path
        self._score_only = score_only
        self._code_model_dim = code_model_dim
        if "code-" not in self.target:
            self._code_model_dim = ""
        self._loader = DataLoader(self._base_path, self.feature, self.target)
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def feature(self):
        return self._feature

    @property
    def target(self):
        return self._target

    @property
    def score(self):
        if not hasattr(self, "_score"):
            raise RuntimeError("Score not set. Need to run.")
        return self._score

    @property
    def null(self):
        if not hasattr(self, "_null"):
            raise RuntimeError("Null not set. Need to run.")
        return self._null

    @property
    def pval(self):
        return (self.score < self.null).sum() / self.null.size

    def _get_fname(self, mode):
        return Path(
            os.path.join(
                self._base_path,
                ".cache",
                "scores",
                self.__class__.__name__.lower(),
                f"{mode}_{self.feature.split('-')[1]}_{self.target.split('-')[1]}{self._code_model_dim}.npy",
            )
        )

    def _set_and_save(self, mode, val, fname):
        setattr(self, f"_{mode}", val)
        np.save(fname, val)
        tag = f": {val:.3f}" if mode == "score" else ""
        self._logger.info(f"Caching '{fname.name}'{tag}.")

    def _run_pipeline(self, mode, iters=1):
        if mode not in ["score", "null"]:
            raise RuntimeError("Mode set incorrectly. Must be 'score' or 'null'")
        fname = self._get_fname(mode)
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
        if fname.exists():
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
    def _run_mapping(self, mode):
        raise NotImplementedError("Handled by subclass.")

    def _plot(self):
        Plotter(self).plot()

    def run(self, iters=1000):
        self._run_pipeline("score")
        if not self._score_only:
            self._run_pipeline("null", iters)
            self._plot()
        return self


class BrainAnalysis(Analysis):
    def __init__(self, feature, target, **kwargs):
        super().__init__(feature, target, **kwargs)

    @property
    def subjects(self):
        return [
            s
            for s in sorted(self._loader.datadir.joinpath("neural_data").glob("*.mat"))
            if "737" not in str(s)  # remove this subject as in Ivanova et al (2020)
        ]

    @abstractmethod
    def _load_subject(self, subject):
        raise NotImplementedError("Handled by subclass.")

    @abstractmethod
    def _shuffle(Y, runs):
        raise NotImplementedError("Handled by subclass.")

    @abstractmethod
    def _calc_score(self, X, Y, runs):
        raise NotImplementedError("Handled by subclass.")

    def _run_mapping(self, mode, cache_subject_scores=True):
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
    def __init__(self, feature, target, **kwargs):
        super().__init__(feature, target, **kwargs)

    @staticmethod
    def _get_metric(Y):
        if Y.ndim == 1:
            metric = ClassificationAccuracy()
        elif Y.ndim == 2:
            if Y.shape[1] == 1:
                metric = PearsonR()
            else:
                metric = RankAccuracy()
        else:
            raise NotImplementedError("Metrics only defined for 1D and 2D arrays.")
        return metric

    def _cross_validate_model(self, X, Y, runs):
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
    def __init__(self, feature, target, **kwargs):
        super().__init__(feature, target, **kwargs)

    def _load_subject(self, subject):
        X, Y, runs = self._loader.get_data(
            self.__class__.__name__.lower(), subject, self._code_model_dim
        )
        return X, Y, runs

    def _shuffle(Y_in, runs):
        if Y_in.shape[0] != runs.shape[0]:
            raise ValueError("Y and runs must have the same number of samples.")
        Y_out = np.zeros(Y_in.shape)
        for run in np.unique(runs):
            Y_out[runs == run] = np.random.permutation(Y_in[runs == run])
        return Y_out

    def _calc_score(self, X, Y, runs):
        score = self._cross_validate_model(X, Y, runs)
        return score


class BrainSimilarity(BrainAnalysis):
    def __init__(self, feature, target, **kwargs):
        super().__init__(feature, target, **kwargs)

    def _load_subject(self, subject):
        X, Y, _ = self._loader.get_data(self.__class__.__name__.lower(), subject)
        return X, Y, _

    def _shuffle(self, Y, _):
        np.random.shuffle(Y)
        return Y

    def _calc_score(self, X, Y, _):
        score = self._similarity_metric(X, Y)
        return score

    @abstractmethod
    def _similarity_metric(self):
        raise NotImplementedError("Handled by subclass.")
