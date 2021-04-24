from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataLoader:
    def __init__(self, network, feature=None):
        self._datadir = Path(__file__).parent.joinpath("inputs", "data_tvals")
        self._events = (12, 6)  # nruns, nblocks
        self._network = network
        self._feature = feature

    @property
    def datadir(self):
        return self._datadir

    @property
    def _runs(self):
        return self._events[0]

    @property
    def _blocks(self):
        return self._events[1]

    @property
    def samples(self):
        return np.prod(self._events)

    def _load_data(self, subject):
        mat = loadmat(subject)
        return (
            mat["data"],
            mat[self._network + "_tags"],
            mat["problem_content"],
            mat["problem_lang"],
            mat["problem_structure"],
        )

    @staticmethod
    def formatcell(matcellarray):
        return np.array([i[0][0] for i in matcellarray])

    def _prep_y(self, content, lang, structure, encoder=LabelEncoder()):
        code = np.array(
            ["sent" if i == "sent" else "code" for i in self.formatcell(lang)]
        )
        if self._feature == "code":
            y = code
            mask = np.ones(code.size, dtype="bool")
        else:
            mask = code == "code"
            if self._feature == "content":
                y = self.formatcell(content)
            elif self._feature == "structure":
                y = self.formatcell(structure)
            else:
                raise LookupError()
        return encoder.fit_transform(y[mask]), mask

    def _prep_x(self, data, parc, mask):
        data = data[:, np.flatnonzero(parc)]
        for i in range(self._runs):
            idx = np.arange(i, self.samples, self._runs)
            data[idx, :] = StandardScaler().fit_transform(data[idx, :])
        return data[mask]

    def _prep_runs(self, mask):
        return np.tile(np.arange(self._runs), self._blocks)[mask]

    def get_xcls(self, subject):  # rsa
        data, parc, content, lang, structure = self._load_data(subject)
        X = self._prep_x(data, parc, np.ones(self.samples, dtype="bool"))
        return X, content, lang, structure

    def get_xyr(self, subject):  # mvpa
        data, parc, content, lang, structure = self._load_data(subject)
        y, mask = self._prep_y(content, lang, structure)
        X = self._prep_x(data, parc, mask)
        runs = self._prep_runs(mask)
        return X, y, runs
