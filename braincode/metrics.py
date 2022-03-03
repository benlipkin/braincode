from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (accuracy_score, mean_squared_error,
                             pairwise_distances)


class Metric(ABC):
    def __init__(self):
        pass

    def __call__(self, X, Y):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if any(y.ndim != 2 for y in [X, Y]):
            raise ValueError("X and Y must be 1D or 2D arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")
        if self.__class__.__name__ != "RepresentationalSimilarity":
            if X.shape[1] != Y.shape[1]:
                raise ValueError("X and Y must have the same number of dimensions.")
        return self._apply_metric(X, Y)

    @abstractmethod
    def _apply_metric(self, X, Y):
        raise NotImplementedError("Handled by subclass.")


class VectorMetric(Metric):
    def __init__(self, reduce=True):
        self._reduce = reduce
        super().__init__()

    def _apply_metric(self, X, Y):
        scores = np.zeros(X.shape[1])
        for i in range(scores.size):
            scores[i] = self._score(X[:, i], Y[:, i])
        if self._reduce:
            return scores.mean()
        return scores

    @abstractmethod
    def _score(self, X, Y):
        raise NotImplementedError("Handled by subclass.")


class MatrixMetric(Metric):
    def __init__(self):
        super().__init__()

    def _apply_metric(self, X, Y):
        score = self._score(X, Y)
        return score

    @abstractmethod
    def _score(self, X, Y):
        raise NotImplementedError("Handled by subclass.")


class PearsonR(VectorMetric):
    @staticmethod
    def _score(x, y):
        r, p = pearsonr(x, y)
        return r


class SpearmanRho(VectorMetric):
    @staticmethod
    def _score(x, y):
        rho, p = spearmanr(x, y)
        return rho


class KendallTau(VectorMetric):
    @staticmethod
    def _score(x, y):
        tau, p = kendalltau(x, y)
        return tau


class RMSE(VectorMetric):
    @staticmethod
    def _score(x, y):
        loss = mean_squared_error(x, y, squared=False)
        return loss


class ClassificationAccuracy(VectorMetric):
    @staticmethod
    def _score(x, y):
        score = accuracy_score(x, y, normalize=True)
        return score


class RankAccuracy(MatrixMetric):
    def __init__(self, distance="euclidean"):
        self._distance = distance
        super().__init__()

    def _score(self, X, Y):
        distances = pairwise_distances(X, Y, metric=self._distance)
        scores = (distances.T > np.diag(distances)).sum(axis=0) / (
            distances.shape[1] - 1
        )
        return scores.mean()


class RepresentationalSimilarity(MatrixMetric):
    def __init__(self, distance="correlation", comparison=PearsonR()):
        self._distance = distance
        self._comparison = comparison

    def _score(self, X, Y):
        X_rdm = pairwise_distances(X, metric=self._distance)
        Y_rdm = pairwise_distances(Y, metric=self._distance)
        if any([m.shape[1] == 1 for m in (X, Y)]):  # can't calc 1D corr dists
            X_rdm[np.isnan(X_rdm)] = 0
            Y_rdm[np.isnan(Y_rdm)] = 0
        indices = np.triu_indices(X_rdm.shape[0], k=1)
        score = self._comparison(X_rdm[indices], Y_rdm[indices])
        return score
