from pathlib import Path

import numpy as np
from features import FeatureExtractor
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataLoader:
    def __init__(self, embedding, feature=None):
        self._datadir = Path(__file__).parent.joinpath("inputs", "neural_data")
        self._events = (12, 6)  # nruns, nblocks
        self._embedding = embedding
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
        if "brain" not in self._embedding:
            raise ValueError(
                "Embedding set incorrectly. Must be brain network to load subject data."
            )
        mat = loadmat(subject)
        return (
            mat["data"],
            mat[self._embedding.split("-")[1] + "_tags"],
            mat["problem_content"],
            mat["problem_lang"],
            mat["problem_structure"],
            mat["problem_ID"],
        )

    @staticmethod
    def formatcell(matcellarray):
        if isinstance(matcellarray[0][0], np.ndarray):
            return np.array([i[0][0] for i in matcellarray])
        elif isinstance(matcellarray[0][0], np.uint8):
            return np.array([i[0] for i in matcellarray])
        else:
            raise TypeError("MATLAB cell array type not handled.")

    def _get_programs(self, lang, id):
        programs = []
        for i in range(id.size):
            fname = list(
                self.datadir.parent.joinpath("python_programs", lang[i]).glob(
                    f"{id[i]}_*"
                )
            )[0].as_posix()
            with open(fname, "r") as f:
                programs.append(f.read())
        return np.array(programs)

    def _prep_y(self, content, lang, structure, id, encoder=LabelEncoder()):
        if self._feature is None:
            raise RuntimeError("Feature attribute not set. Need to properly init.")
        code = np.array(
            ["sent" if i == "sent" else "code" for i in self.formatcell(lang)]
        )
        if self._feature == "task-code":
            y = code
            mask = np.ones(code.size, dtype="bool")
        else:
            mask = code == "code"
            if self._feature in ["task-content", "task-structure"]:
                y = self.formatcell(locals()[self._feature.split("-")[1]])[mask]
            elif self._feature in ["code-bow", "code-tfidf"]:  # returns dense features
                y = self._get_programs(
                    self.formatcell(lang)[mask], self.formatcell(id)[mask]
                )
                encoder = FeatureExtractor(self._feature)
            else:
                raise ValueError("Feature not recognized. Select valid feature.")
        return encoder.fit_transform(y), mask

    def _prep_x(self, data, parc, mask):
        data = data[:, np.flatnonzero(parc)]
        for i in range(self._runs):
            idx = np.arange(i, self.samples, self._runs)
            data[idx, :] = StandardScaler().fit_transform(data[idx, :])
        return data[mask]

    def _prep_runs(self, mask):
        return np.tile(np.arange(self._runs), self._blocks)[mask]

    def get_xcls(self, subject):  # rsa
        data, parc, content, lang, structure, id = self._load_data(subject)
        X = self._prep_x(data, parc, np.ones(self.samples, dtype="bool"))
        return X, content, lang, structure

    def get_xyr(self, subject):  # mvpa
        data, parc, content, lang, structure, id = self._load_data(subject)
        y, mask = self._prep_y(content, lang, structure, id)
        X = self._prep_x(data, parc, mask)
        runs = self._prep_runs(mask)
        return X, y, runs

    def get_xy(self):  # prda
        raise NotImplementedError()  # resume work here
