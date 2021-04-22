import os

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from util import formatcell, get_mat, parse_mat


class DataLoader:
    def __init__(self):
        self.__datadir = os.path.join(os.path.dirname(__file__), "inputs", "data_tvals")

    @property
    def datadir(self):
        return self.__datadir

    def load_data(self, subject, network):
        return parse_mat(get_mat(os.path.join(self.datadir, subject)), network)

    @staticmethod
    def prep_y(content, lang, structure, feature, encoder=LabelEncoder()):
        code = np.array(["sent" if i == "sent" else "code" for i in formatcell(lang)])
        if feature == "sent v code":
            y = code
            idx = np.ones(code.size, dtype="bool")
        else:
            idx = code == "code"
            if feature == "math v str":
                y = formatcell(content)
            elif feature == "seq v for v if":
                y = formatcell(structure)
            else:
                raise LookupError()
        return encoder.fit_transform(y[idx]), idx

    @staticmethod
    def prep_x(data, parc):
        data = data[:, np.flatnonzero(parc)]
        for i in range(12):
            idx = np.arange(i, 72, 12)
            data[idx, :] = StandardScaler().fit_transform(data[idx, :])
        return data

    @staticmethod
    def get_runs():
        return np.tile(np.arange(12), 6)

    def get_xyr(self, subject, network, feature):
        data, parc, content, lang, structure = self.load_data(subject, network)
        y, idx = self.prep_y(content, lang, structure, feature)
        X = self.prep_x(data, parc)[idx]
        runs = self.get_runs()[idx]
        return X, y, runs
