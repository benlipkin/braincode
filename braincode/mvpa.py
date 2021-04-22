from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import LinearSVC
from tqdm import tqdm
from util import accuracy


class MVPA:
    def __init__(self, network, feature):
        self.__loader = DataLoader()
        self.__network = network
        self.__feature = feature
        self.__score = None
        self.__null = None

    @property
    def network(self):
        return self.__network

    @property
    def feature(self):
        return self.__feature

    @property
    def score(self):
        return self.__score

    @property
    def null(self):
        return self.__null

    @property
    def pval(self):
        return (self.score < self.null).sum() / self.null.size

    @score.setter
    def score(self, value):
        self.__score = value

    @null.setter
    def null(self, value):
        self.__null = value

    @staticmethod
    def __shuffle_within_runs(y_in, runs):
        y_out = np.zeros(y_in.shape)
        for run in np.unique(runs):
            y_out[runs == run] = np.random.permutation(y_in[runs == run])
        return y_out

    @staticmethod
    def __cross_validate_model(X, y, runs):
        classes = np.unique(y)
        classifier = LinearSVC(max_iter=1e5)
        cmat = np.zeros((classes.size, classes.size))
        for train, test in LeaveOneGroupOut().split(X, y, runs):
            model = classifier.fit(X[train], y[train])
            cmat += confusion_matrix(y[test], model.predict(X[test]), labels=classes)
        return cmat

    def __run_mvpa(self, mode):
        for subject in sorted(self.__loader.datadir.iterdir()):
            X, y, runs = self.__loader.get_xyr(subject, self.network, self.feature)
            if mode == "null":
                y = self.__shuffle_within_runs(y, runs)
            cv_results = self.__cross_validate_model(X, y, runs)
            cmat = cmat + cv_results if "cmat" in locals() else cv_results
        return cmat

    def __run_pipeline(self, mode, iters=1):
        assert mode in ["score", "null"]
        fname = Path(__file__).parent.joinpath(
            "outputs", f"{mode}_{'_'.join(self.feature.split())}_{self.network}.npy"
        )
        if fname.exists():
            setattr(self, mode, np.load(fname, allow_pickle=True))
            return
        samples = np.zeros((iters))
        for idx in tqdm(range(iters)):
            cmat = self.__run_mvpa(mode)
            if mode == "score":
                self.score = accuracy(cmat)
                np.save(fname, self.score)
                return
            samples[idx] = accuracy(cmat)
        self.null = samples
        np.save(fname, self.null)

    def __plot_results(self):
        plt.hist(self.null, bins=25, color="lightblue", edgecolor="black")
        plt.axvline(self.score, color="black", linewidth=3)
        plt.xlim(
            {
                "sent v code": [0.4, 0.9],
                "math v str": [0.4, 0.65],
                "seq v for v if": [0.25, 0.55],
            }[self.feature]
        )
        plt.savefig(
            Path(__file__).parent.joinpath(
                "plots", "mvpa", f"{'_'.join(self.feature.split())}_{self.network}.png"
            )
        )
        plt.clf()

    def run(self, perms=True, iters=1000):
        self.__run_pipeline("score")
        if perms:
            self.__run_pipeline("null", iters)
            self.__plot_results()
        return self
