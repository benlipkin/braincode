import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from data import DataLoader
from plots import Plotter
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import pairwise_distances
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
        self._logger.info(f"Caching '{fname.name}'.")

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
            score = self._run_decoding(mode)
            if mode == "score":
                self._set_and_save(mode, score, fname)
                return
            samples[idx] = score
        self._set_and_save(mode, samples, fname)

    @abstractmethod
    def _run_decoding(self, mode):
        raise NotImplementedError("Handled by subclass.")

    def _plot(self):
        Plotter(self).plot()

    def run(self, iters=1000):
        self._run_pipeline("score")
        if not self._score_only:
            self._run_pipeline("null", iters)
            self._plot()
        return self


class Decoder(Analysis):
    def __init__(self, feature, target, base_path, score_only, code_model_dim):
        super().__init__(feature, target, base_path, score_only, code_model_dim)

    @staticmethod
    def _shuffle_within_runs(y_in, runs):
        y_out = np.zeros(y_in.shape)
        for run in np.unique(runs):
            y_out[runs == run] = np.random.permutation(y_in[runs == run])
        return y_out

    @staticmethod
    def _rank_accuracy(pred, true, metric="euclidean"):
        distances = pairwise_distances(pred, true, metric=metric)
        scores = (distances.T > np.diag(distances)).sum(axis=0) / (
            distances.shape[1] - 1
        )
        return scores.mean()

    def _cross_validate_model(self, X, y, runs):
        model_class = RidgeClassifierCV if y.ndim == 1 else RidgeCV
        scores = np.zeros(np.unique(runs).size)
        for idx, (train, test) in enumerate(LeaveOneGroupOut().split(X, y, runs)):
            model = model_class(alphas=np.logspace(-2, 2, 9)).fit(X[train], y[train])
            if y.ndim == 1:
                scores[idx] = model.score(X[test], y[test])
            else:
                if y.shape[1] == 1:
                    scores[idx] = np.corrcoef(
                        model.predict(X[test]).squeeze(), y[test].squeeze()
                    )[1, 0]
                else:
                    scores[idx] = self._rank_accuracy(model.predict(X[test]), y[test])
        return scores.mean()


class MVPA(Decoder):
    def __init__(self, feature, target, base_path, score_only, code_model_dim):
        super().__init__(feature, target, base_path, score_only, code_model_dim)

    def _run_decoding(self, mode, cache_subject_scores=True):
        subjects = sorted(self._loader.datadir.joinpath("neural_data").glob("*.mat"))
        subjects = [
            s for s in subjects if "737" not in str(s)
        ]  # remove this subject as in Ivanova et al (2020)
        scores = np.zeros(len(subjects))
        for idx, subject in enumerate(subjects):
            X, y, runs = self._loader.get_data(
                self.__class__.__name__.lower(), subject, self._code_model_dim
            )
            if mode == "null":
                y = self._shuffle_within_runs(y, runs)
            scores[idx] = self._cross_validate_model(X, y, runs)
        if mode == "score" and cache_subject_scores:
            temp_mode = "subjects"
            self._set_and_save(temp_mode, scores, self._get_fname(temp_mode))
        return scores.mean()


class PRDA(Decoder):
    def __init__(self, feature, target, base_path, score_only, code_model_dim):
        super().__init__(feature, target, base_path, score_only, code_model_dim)

    def _run_decoding(self, mode):
        X, y, runs = self._loader.get_data(self.__class__.__name__.lower())
        if mode == "null":
            np.random.shuffle(y)
        return self._cross_validate_model(X, y, runs)
