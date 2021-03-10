import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from braincode_mvpa import memoize, get_mat, parse_mat, formatcell


class correlation_matrix:
    def __init__(self):
        self.matrix = np.zeros((72, 72), dtype="float64")
        self.axes = np.empty((72, 3), dtype="U4")
        self.subjects = 0

    def _axes_empty(self):
        return "" in self.axes

    def _set_axis(self, index, array):
        self.axes[:, index] = array

    def _update_coef(self, data, parc):
        self.matrix += np.corrcoef(data[:, np.flatnonzero(parc)])
        self.subjects += 1

    def get_axes(self):
        return self.axes

    def get_coef(self):
        return self.matrix / self.subjects

    def add_subject(self, fname, network):
        data, parc, content, lang, structure = parse_mat(get_mat(fname), network)
        data = StandardScaler().fit_transform(data.T).T
        self._update_coef(data, parc)
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
    networks = ["lang", "MD", "code"]
    for network in networks:
        corr = correlation_matrix()
        for subject in sorted(os.listdir(input_dir)):
            corr.add_subject(input_dir + subject, network)
        corr.plot("../outputs/%s_corr.jpg" % (network))


if __name__ == "__main__":
    main()
