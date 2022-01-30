import os
import pickle as pkl
from functools import lru_cache
from pathlib import Path

import numpy as np
from benchmarks import ProgramBenchmark
from embeddings import ProgramEmbedder
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataLoader:
    def __init__(self, base_path, feature, target):
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

    @staticmethod
    def _get_joint_network_indices(network, mat):
        networks = network.split("+")
        if len(networks) == 2:
            network_indices = mat[f"{networks[0]}_tags"] + mat[f"{networks[1]}_tags"]
        else:
            raise ValueError("Network not supported. Select valid network.")
        network_indices[network_indices > 1] = 1
        return network_indices

    def _load_brain_data(self, subject):
        if "brain" not in self._feature:
            raise ValueError(
                "Feature set incorrectly. Must be brain network to load subject data."
            )
        mat = loadmat(subject)
        network = self._feature.split("-")[1]
        if "+" in network:
            network_indices = self._get_joint_network_indices(network, mat)
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
        programs, fnames = [], []
        for i in range(id.size):
            fnames.append(
                list(
                    self.datadir.joinpath("python_programs", lang[i]).glob(f"{id[i]}_*")
                )[0].as_posix()
            )
            with open(fnames[-1], "r") as f:
                programs.append(f.read())
        return np.array(programs), np.array(fnames)

    def _prep_y(
        self, content, lang, structure, id, code_model_dim, encoder=LabelEncoder()
    ):
        code = np.array(
            ["sent" if i == "sent" else "code" for i in self._formatcell(lang)]
        )
        if self._target == "test-code":
            y = code
            mask = np.ones(code.size, dtype="bool")
        else:
            mask = code == "code"
            if self._target in ["task-content", "test-lang", "task-structure"]:
                y = self._formatcell(locals()[self._target.split("-")[1]])[mask]
            else:
                y, fnames = self._load_select_programs(
                    self._formatcell(lang)[mask], self._formatcell(id)[mask]
                )
                if self._target in [
                    "code-projection",
                    "code-bow",
                    "code-tfidf",
                    "code-seq2seq",
                    "code-transformer",
                    "code-xlnet",
                    "code-bert",
                    "code-gpt2",
                    "code-roberta",
                    # "code-ada",
                    # "code-babbage",
                ]:
                    encoder = ProgramEmbedder(
                        self._target, self._base_path, code_model_dim
                    )
                elif self._target in [
                    "task-lines",
                    "task-bytes",
                    "task-nodes",
                    "task-tokens",
                    "task-halstead",
                    "task-cyclomatic",
                ]:
                    encoder = ProgramBenchmark(self._target, self._base_path, fnames)
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
        programs, content, lang, structure, fnames = [], [], [], [], []
        files = list(self.datadir.joinpath("python_programs", "en").rglob("*.py"))
        for file in sorted(files):
            fnames.append(file.as_posix())
            with open(fnames[-1], "r") as f:
                programs.append(f.read())
            info = fnames[-1].split(os.sep)[-1].split(" ")[1].split("_")
            content.append(info[0])
            lang.append(fnames[-1].split(os.sep)[-2])
            structure.append(info[1])
        return (
            np.array(programs),
            np.array(content),
            np.array(lang),
            np.array(structure),
            np.array(fnames),
        )

    def _calc_data_mvpa(self, subject, code_model_dim):
        data, parc, content, lang, structure, id = self._load_brain_data(subject)
        y, mask = self._prep_y(content, lang, structure, id, code_model_dim)
        X = self._prep_x(data, parc, mask)
        runs = self._prep_runs(self._runs, self._blocks)[mask]
        return X, y, runs

    def _calc_data_prda(self, k=5):
        programs, content, lang, structure, fnames = self._load_all_programs()
        if self._target in ["task-content", "task-structure"]:
            y = locals()[self._target.split("-")[1]]
        else:
            y = ProgramBenchmark(self._target, self._base_path, fnames).fit_transform(
                programs
            )
        X = ProgramEmbedder(self._feature, self._base_path, "").fit_transform(programs)
        runs = self._prep_runs(k, (y.size // k + 1))[: y.size]  # kfold CV
        return X, y, runs

    def _get_fname(self, analysis, subject="", code_model_dim=""):
        if subject != "":
            subject = subject.name.split(".")[0]
        if code_model_dim != "":
            code_model_dim = f"_dim{code_model_dim}"
        fname = Path(
            os.path.join(
                self._base_path,
                ".cache",
                "representations",
                analysis,
                f"{self._feature.split('-')[1]}_{self._target.split('-')[1]}{subject}{code_model_dim}.pkl",
            )
        )
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
        return fname

    @lru_cache(maxsize=None)
    def get_data(self, analysis, subject="", code_model_dim=""):
        fname = self._get_fname(analysis, subject, code_model_dim)
        if fname.exists():
            with open(fname, "rb") as f:
                data = pkl.load(f)
            return data["X"], data["y"], data["runs"]
        else:
            if analysis in ["mvpa", "rsa"]:
                X, y, runs = self._calc_data_mvpa(subject, code_model_dim)
            elif analysis == "prda":
                X, y, runs = self._calc_data_prda()
            with open(fname, "wb") as f:
                pkl.dump({"X": X, "y": y, "runs": runs}, f)
            return X, y, runs
