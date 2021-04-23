from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataLoader:
    def __init__(self, network, feature=None):
        self.__datadir = Path(__file__).parent.joinpath("inputs", "data_tvals")
        self.__events = (12, 6)  # nruns, nblocks
        self.__network = network
        self.__feature = feature

    @property
    def datadir(self):
        return self.__datadir

    @property
    def __runs(self):
        return self.__events[0]

    @property
    def __blocks(self):
        return self.__events[1]

    @property
    def samples(self):
        return np.prod(self.__events)

    def __load_data(self, subject):
        mat = loadmat(subject)
        return (
            mat["data"],
            mat[self.__network + "_tags"],
            mat["problem_content"],
            mat["problem_lang"],
            mat["problem_structure"],
        )

    @staticmethod
    def formatcell(matcellarray):
        return np.array([i[0][0] for i in matcellarray])

    def __prep_y(self, content, lang, structure, encoder=LabelEncoder()):
        code = np.array(
            ["sent" if i == "sent" else "code" for i in self.formatcell(lang)]
        )
        if self.__feature == "code":
            y = code
            mask = np.ones(code.size, dtype="bool")
        else:
            mask = code == "code"
            if self.__feature == "content":
                y = self.formatcell(content)
            elif self.__feature == "structure":
                y = self.formatcell(structure)
            else:
                raise LookupError()
        return encoder.fit_transform(y[mask]), mask

    def __prep_x(self, data, parc, mask):
        data = data[:, np.flatnonzero(parc)]
        for i in range(self.__runs):
            idx = np.arange(i, self.samples, self.__runs)
            data[idx, :] = StandardScaler().fit_transform(data[idx, :])
        return data[mask]

    def __prep_runs(self, mask):
        return np.tile(np.arange(self.__runs), self.__blocks)[mask]

    def get_xcls(self, subject):  # rsa
        data, parc, content, lang, structure = self.__load_data(subject)
        X = self.__prep_x(data, parc, np.ones(self.samples, dtype="bool"))
        return X, content, lang, structure

    def get_xyr(self, subject):  # mvpa
        data, parc, content, lang, structure = self.__load_data(subject)
        y, mask = self.__prep_y(content, lang, structure)
        X = self.__prep_x(data, parc, mask)
        runs = self.__prep_runs(mask)
        return X, y, runs
