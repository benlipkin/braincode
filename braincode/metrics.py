from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (accuracy_score, mean_squared_error,
                             pairwise_distances)


class Metric(ABC):
    def __init__(self):
        pass

    def __call__(self, Y_pred, Y_true):
        if Y_pred.ndim == 1:
            Y_pred = Y_pred.reshape(-1, 1)
        if Y_true.ndim == 1:
            Y_true = Y_true.reshape(-1, 1)
        if any(y.ndim != 2 for y in [Y_pred, Y_true]):
            raise ValueError("Y_pred and Y_true must be 1D or 2D arrays.")
        if not Y_pred.shape[0] == Y_true.shape[0]:
            raise ValueError("Y_pred and Y_true must have the same number of samples.")
        if not Y_pred.shape[1] == Y_true.shape[1]:
            raise ValueError(
                "Y_pred and Y_true must have the same number of dimensions."
            )
        return self._apply_metric(Y_pred, Y_true)

    @abstractmethod
    def _apply_metric(self, Y_pred, Y_true):
        raise NotImplementedError("Handled by subclass.")


class VectorMetric(Metric):
    def __init__(self, reduce=True):
        self._reduce = reduce
        super().__init__()

    def _apply_metric(self, Y_pred, Y_true):
        scores = np.zeros(Y_pred.shape[1])
        for i in range(scores.size):
            scores[i] = self._score(Y_pred[:, i], Y_true[:, i])
        if self._reduce:
            return scores.mean()
        return scores

    @abstractmethod
    def _score(self, y_pred, y_true):
        raise NotImplementedError("Handled by subclass.")


class MatrixMetric(Metric):
    def __init__(self):
        super().__init__()

    def _apply_metric(self, Y_pred, Y_true):
        score = self._score(Y_pred, Y_true)
        return score

    @abstractmethod
    def _score(self, y_pred, y_true):
        raise NotImplementedError("Handled by subclass.")


class PearsonR(VectorMetric):
    @staticmethod
    def _score(y_pred, y_true):
        r, p = pearsonr(y_pred, y_true)
        return r


class SpearmanRho(VectorMetric):
    @staticmethod
    def _score(y_pred, y_true):
        rho, p = spearmanr(y_pred, y_true)
        return rho


class KendallTau(VectorMetric):
    @staticmethod
    def _score(y_pred, y_true):
        tau, p = kendalltau(y_pred, y_true)
        return tau


class RMSE(VectorMetric):
    @staticmethod
    def _score(y_pred, y_true):
        loss = mean_squared_error(y_pred, y_true, squared=False)
        return loss


class ClassificationAccuracy(VectorMetric):
    @staticmethod
    def _score(y_pred, y_true):
        score = accuracy_score(y_true, y_pred, normalize=True)
        return score


class RankAccuracy(MatrixMetric):
    def __init__(self, distance="euclidean"):
        self._distance = distance
        super().__init__()

    def _score(self, Y_pred, Y_true):
        distances = pairwise_distances(Y_pred, Y_true, metric=self._distance)
        scores = (distances.T > np.diag(distances)).sum(axis=0) / (
            distances.shape[1] - 1
        )
        return scores.mean()
