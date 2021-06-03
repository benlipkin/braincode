from pathlib import Path

import numpy as np
from data import DataLoader
from plots import Plotter
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm


class MVPA:
    def __init__(self, network, feature):
        self._network = network
        self._feature = feature
        self._loader = DataLoader(self.network, self.feature)
        self._score = None
        self._null = None

    @property
    def network(self):
        return self._network

    @property
    def feature(self):
        return self._feature

    @property
    def score(self):
        return self._score

    @property
    def null(self):
        return self._null

    @property
    def pval(self):
        return (self.score < self.null).sum() / self.null.size

    @staticmethod
    def _shuffle_within_runs(y_in, runs):
        y_out = np.zeros(y_in.shape)
        for run in np.unique(runs):
            y_out[runs == run] = np.random.permutation(y_in[runs == run])
        return y_out

    @staticmethod
    def _rank_accuracy(pred, true, metric="cosine"):
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
                scores[idx] = self._rank_accuracy(model.predict(X[test]), y[test])
        return scores.mean()

    def _run_mvpa(self, mode):
        subjects = sorted(self._loader.datadir.iterdir())
        scores = np.zeros(len(subjects))
        for idx, subject in enumerate(subjects):
            X, y, runs = self._loader.get_xyr(subject)
            if mode == "null":
                y = self._shuffle_within_runs(y, runs)
            scores[idx] = self._cross_validate_model(X, y, runs)
        return scores.mean()

    def _run_pipeline(self, mode, iters=1):
        assert mode in ["score", "null"]
        fname = Path(__file__).parent.joinpath(
            "outputs", f"{mode}_{self.feature}_{self.network}.npy"
        )
        if fname.exists():
            setattr(self, "_" + mode, np.load(fname, allow_pickle=True))
            return
        samples = np.zeros((iters))
        for idx in tqdm(range(iters)):
            score = self._run_mvpa(mode)
            if mode == "score":
                self._score = score
                np.save(fname, self.score)
                return
            samples[idx] = score
        self._null = samples
        np.save(fname, self.null)

    def _plot(self):
        Plotter(self).plot()

    def run(self, perms=True, iters=1000):
        self._run_pipeline("score")
        if perms:
            self._run_pipeline("null", iters)
            self._plot()
        return self
