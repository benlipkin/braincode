from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from util import formatcell, get_mat, parse_mat


class DataLoader:
    def __init__(self):
        self.__datadir = Path(__file__).parent.joinpath("inputs", "data_tvals")
        self.__events = (12, 6)  # ntrials, nruns

    @property
    def datadir(self):
        return self.__datadir

    @property
    def runs(self):
        return self.__events[0]

    @property
    def trials(self):
        return self.__events[1]

    @property
    def samples(self):
        return np.prod(self.__events)

    def load_data(self, subject, network):
        return parse_mat(get_mat(subject), network)

    def prep_y(self, content, lang, structure, feature, encoder=LabelEncoder()):
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

    def prep_x(self, data, parc):
        data = data[:, np.flatnonzero(parc)]
        for i in range(self.runs):
            idx = np.arange(i, self.samples, self.runs)
            data[idx, :] = StandardScaler().fit_transform(data[idx, :])
        return data

    def get_runs(self):
        return np.tile(np.arange(self.runs), self.trials)

    def get_xyr(self, subject, network, feature):
        data, parc, content, lang, structure = self.load_data(subject, network)
        y, idx = self.prep_y(content, lang, structure, feature)
        X = self.prep_x(data, parc)[idx]
        runs = self.get_runs()[idx]
        return X, y, runs
