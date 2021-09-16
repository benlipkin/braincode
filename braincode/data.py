import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from encoding import ProgramEncoder
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataLoader:
    def __init__(self, base_path, feature, target=None):
        self._datadir = Path(os.path.join(base_path, "inputs"))
        self._events = (12, 6)  # nruns, nblocks
        self._feature = feature
        self._target = target
        self._base_path = base_path

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

    def _load_brain_data(self, subject):
        if "brain" not in self._feature:
            raise ValueError(
                "Feature set incorrectly. Must be brain network to load subject data."
            )
        mat = loadmat(subject)
        network = self._feature.split("-")[1]
        if network == "composite":
            network_indices = (
                mat["MD_tags"] + mat["lang_tags"] + mat["vis_tags"] + mat["aud_tags"]
            )
            network_indices[network_indices > 1] = 1
        else:
            network_indices = mat[f"{network}_tags"]
        return (
            mat["data"],
            network_indices,
            mat["problem_content"],
            mat["problem_lang"],
            mat["problem_structure"],
            mat["problem_ID"],
        )

    @staticmethod
    def _formatcell(matcellarray):
        if isinstance(matcellarray[0][0], np.ndarray):
            return np.array([i[0][0] for i in matcellarray])
        elif isinstance(matcellarray[0][0], np.uint8):
            return np.array([i[0] for i in matcellarray])
        else:
            raise TypeError("MATLAB cell array type not handled.")

    def _load_select_programs(self, lang, id):
        programs = []
        for i in range(id.size):
            fname = list(
                self.datadir.joinpath("python_programs", lang[i]).glob(f"{id[i]}_*")
            )[0].as_posix()
            with open(fname, "r") as f:
                programs.append(f.read())
        return np.array(programs)

    def _prep_y(self, content, lang, structure, id, encoder=LabelEncoder()):
        if self._target is None:
            raise RuntimeError("Target attribute not set. Need to properly init.")
        code = np.array(
            ["sent" if i == "sent" else "code" for i in self._formatcell(lang)]
        )
        if self._target == "test-code":
            y = code
            mask = np.ones(code.size, dtype="bool")
        else:
            mask = code == "code"
            if self._target in ["task-content", "task-lang", "task-structure"]:
                y = self._formatcell(locals()[self._target.split("-")[1]])[mask]
            elif self._target in [
                "code-random",
                "code-bow",
                "code-tfidf",
                "code-seq2seq",
                "code-xlnet",
                "code-ct",
                "code-codeberta",
            ]:
                y = self._load_select_programs(
                    self._formatcell(lang)[mask], self._formatcell(id)[mask]
                )
                encoder = ProgramEncoder(self._target, self._base_path)
            else:
                raise ValueError("Target not recognized. Select valid target.")
        return encoder.fit_transform(y), mask

    def _prep_x(self, data, parc, mask):
        data = data[:, np.flatnonzero(parc)]
        for i in range(self._runs):
            idx = np.arange(i, self.samples, self._runs)
            data[idx, :] = StandardScaler().fit_transform(data[idx, :])
        return data[mask]

    def _prep_runs(self, runs, blocks):
        return np.tile(np.arange(runs), blocks)

    def _load_all_programs(self):
        programs, content, lang, structure = [], [], [], []
        files = list(self.datadir.joinpath("python_programs").rglob("*.py"))
        for file in sorted(files):
            fname = file.as_posix()
            with open(fname, "r") as f:
                programs.append(f.read())
            info = fname.split(os.sep)[-1].split(" ")[1].split("_")
            content.append(info[0])
            lang.append(fname.split(os.sep)[-2])
            structure.append(info[1])
        return (
            np.array(programs),
            np.array(content),
            np.array(lang),
            np.array(structure),
        )

    def get_data_rsa(self, subject):
        data, parc, content, lang, structure, id = self._load_brain_data(subject)
        X = self._prep_x(data, parc, np.ones(self.samples, dtype="bool"))
        axes = np.vstack([self._formatcell(ar) for ar in [content, lang, structure]]).T
        return X, axes

    @lru_cache(maxsize=None)
    def get_data_mvpa(self, subject):
        data, parc, content, lang, structure, id = self._load_brain_data(subject)
        y, mask = self._prep_y(content, lang, structure, id)
        X = self._prep_x(data, parc, mask)
        runs = self._prep_runs(self._runs, self._blocks)[mask]
        return X, y, runs

    @lru_cache(maxsize=None)
    def get_data_prda(self, k=5):
        programs, content, lang, structure = self._load_all_programs()
        y = locals()[self._target.split("-")[1]]
        X = ProgramEncoder(self._feature, self._base_path).fit_transform(programs)
        runs = self._prep_runs(k, (y.size // k + 1))[: y.size]  # kfold CV
        return X, y, runs
