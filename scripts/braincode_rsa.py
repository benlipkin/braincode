import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from braincode_util import *


class correlation_matrix:
    def __init__(self):
        self._matrix = np.zeros((72, 72), dtype="float64")
        self._axes = np.empty((72, 3), dtype="U4")
        self._subjects = 0

    def _axes_empty(self):
        return "" in self._axes

    def _set_axis(self, index, array):
        self._axes[:, index] = array

    def _update_coef(self, data):
        self._matrix += np.corrcoef(data)
        self._subjects += 1

    def get_axes(self):
        return self._axes

    def get_coef(self):
        return self._matrix / self._subjects

    def add_subject(self, fname, network):
        data, parc, content, lang, structure = parse_mat(get_mat(fname), network)
        self._update_coef(prep_x(data, parc))
        if self._axes_empty():
            self._set_axis(0, formatcell(content))
            self._set_axis(1, formatcell(lang))
            self._set_axis(2, formatcell(structure))

    def plot(self, fname, show=False):
        corr_coef = self.get_coef()
        axes = self.get_axes()
        ticks = np.arange(corr_coef.shape[0])
        labels = np.array(["_".join(row) for row in axes])
        indices = np.argsort(labels)
        plt.imshow(corr_coef[indices, :][:, indices])
        plt.xticks(ticks, labels[indices], fontsize=5, rotation=90)
        plt.yticks(ticks, labels[indices], fontsize=5)
        plt.clim([0, 1])
        plt.colorbar()
        plt.savefig(fname)
        plt.show() if show else plt.clf()


def main():
    input_dir = "../inputs/item_data_tvals_20201002/"
    networks = ["lang", "MD", "aud"]
    for network in networks:
        corr = correlation_matrix()
        for subject in sorted(os.listdir(input_dir)):
            corr.add_subject(input_dir + subject, network)
        corr.plot("../plots/corr/%s.jpg" % (network))


if __name__ == "__main__":
    main()
