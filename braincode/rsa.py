import os

import matplotlib.pyplot as plt
import numpy as np
from data import DataLoader
from util import formatcell


class RSA:
    def __init__(self, network):
        self.__network = network
        self.__corr = None

    @property
    def network(self):
        return self.__network

    @property
    def corr(self):
        return self.__corr

    def __calc_corr(self):
        self.__corr = CorrelationMatrix()
        for subject in sorted(os.listdir(self.corr.loader.datadir)):
            self.corr.add_subject(subject, self.network)

    def __plot_corr(self):
        self.corr.plot(
            os.path.join(
                os.path.dirname(__file__),
                "plots",
                "rsa",
                f"{self.network}.jpg",
            )
        )

    def run(self):
        self.__calc_corr()
        self.__plot_corr()
        return self


class CorrelationMatrix:
    def __init__(self):
        self.__loader = DataLoader()
        self.__matrix = np.zeros((72, 72), dtype="float64")
        self.__axes = np.empty((72, 3), dtype="U4")
        self.__subjects = 0

    @property
    def loader(self):
        return self.__loader

    @property
    def axes(self):
        return self.__axes

    @property
    def coef(self):
        return self.__matrix / self.__subjects

    def __axes_empty(self):
        return "" in self.__axes

    def __set_axis(self, index, array):
        self.__axes[:, index] = array

    def __update_coef(self, data):
        self.__matrix += np.corrcoef(data)
        self.__subjects += 1

    def add_subject(self, subject, network):
        data, parc, content, lang, structure = self.loader.load_data(subject, network)
        self.__update_coef(self.loader.prep_x(data, parc))
        if self.__axes_empty():
            self.__set_axis(0, formatcell(content))
            self.__set_axis(1, formatcell(lang))
            self.__set_axis(2, formatcell(structure))

    def plot(self, fname, show=False):
        ticks = np.arange(self.coef.shape[0])
        labels = np.array(["_".join(row) for row in self.axes])
        indices = np.argsort(labels)
        plt.imshow(self.coef[indices, :][:, indices])
        plt.xticks(ticks, labels[indices], fontsize=5, rotation=90)
        plt.yticks(ticks, labels[indices], fontsize=5)
        plt.clim([0, 1])
        plt.colorbar()
        plt.savefig(fname)
        plt.show() if show else plt.clf()
