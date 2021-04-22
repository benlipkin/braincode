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

    @corr.setter
    def corr(self, value):
        self.__corr = value

    def __calc_corr(self):
        corr = CorrelationMatrix()
        for subject in sorted(os.listdir(DataLoader().datadir)):
            corr.add_subject(subject, self.network)
        self.corr = corr

    def __plot_corr(self):
        self.corr.plot(
            os.path.join(
                os.path.dirname(__file__),
                "plots",
                "corr",
                f"{self.network}.jpg",
            )
        )

    def run(self):
        self.__calc_corr()
        self.__plot_corr()
        return self


class CorrelationMatrix:
    def __init__(self):
        self.__matrix = np.zeros((72, 72), dtype="float64")
        self.__axes = np.empty((72, 3), dtype="U4")
        self.__subjects = 0

    @property
    def matrix(self):
        return self.__matrix

    @property
    def axes(self):
        return self.__axes

    @property
    def subjects(self):
        return self.__subjects

    @property
    def coef(self):
        return self.matrix / self.subjects

    @matrix.setter
    def matrix(self, value):
        self.__matrix = value

    @subjects.setter
    def subjects(self, value):
        self.__subjects = value

    def __axes_empty(self):
        return "" in self.axes

    def __set_axis(self, index, array):
        self.__axes[:, index] = array

    def __update_coef(self, data):
        self.matrix += np.corrcoef(data)
        self.subjects += 1

    def add_subject(self, subject, network):
        data, parc, content, lang, structure = DataLoader().load_data(subject, network)
        self.__update_coef(DataLoader().prep_x(data, parc))
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
