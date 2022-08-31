import os
import pickle as pkl
import typing
from abc import abstractmethod
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from braincode.abstract import Object
from braincode.benchmarks import ProgramBenchmark
from braincode.embeddings import ProgramEmbedder


class DataLoader(Object):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._datadir = self._base_path.joinpath("inputs")
        self._events = (12, 6)  # nruns, nblocks

    @property
    def datadir(self) -> Path:
        return self._datadir

    @property
    def _runs(self) -> int:
        return self._events[0]

    @property
    def _blocks(self) -> int:
        return self._events[1]

    @property
    def samples(self) -> int:
        return int(np.prod(self._events))

    def _load_brain_data(
        self, subject: Path
    ) -> typing.Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        if "brain" not in self._feature:
            raise ValueError(
                "Feature set incorrectly. Must be brain network to load subject data."
            )
        mat = loadmat(subject)
        network = self._feature.split("-")[1]
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
    def _formatcell(matcellarray: np.ndarray) -> np.ndarray:
        if isinstance(matcellarray[0][0], np.ndarray):
            return np.array([i[0][0] for i in matcellarray])
        if isinstance(matcellarray[0][0], np.uint8):
            return np.array([i[0] for i in matcellarray])
        raise TypeError("MATLAB cell array type not handled.")

    def _load_select_programs(
        self, lang: np.ndarray, ident: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        programs, fnames = [], []
        for i in range(ident.size):
            fnames.append(
                list(
                    self.datadir.joinpath("python_programs", lang[i]).glob(
                        f"{ident[i]}_*"
                    )
                )[0].as_posix()
            )
            with open(fnames[-1], "r") as f:
                programs.append(f.read())
        return np.array(programs), np.array(fnames)

    def _prep_code_reps(
        self,
        content: np.ndarray,
        lang: np.ndarray,
        structure: np.ndarray,
        ident: np.ndarray,
        encoder=LabelEncoder(),
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        code = np.array(
            ["sent" if i == "sent" else "code" for i in self._formatcell(lang)]
        )
        if self._target == "test-code":
            Y = code
            mask = np.ones(code.size, dtype="bool")
        else:
            mask = code == "code"
            if self._target in ["task-content", "test-lang", "task-structure"]:
                Y = self._formatcell(locals()[self._target.split("-")[1]])[mask]
            else:
                Y, fnames = self._load_select_programs(
                    self._formatcell(lang)[mask], self._formatcell(ident)[mask]
                )
                if "code-" in self._target:
                    encoder = ProgramEmbedder(
                        self._target, self._base_path, self._code_model_dim
                    )
                elif "task-" in self._target:
                    encoder = ProgramBenchmark(self._target, self._base_path, fnames)
                else:
                    raise ValueError("Target not recognized. Select valid target.")
        return encoder.fit_transform(Y), mask

    def _prep_brain_reps(
        self, data: np.ndarray, parc: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        data = data[:, np.flatnonzero(parc)]
        for i in range(self._runs):
            idx = np.arange(i, self.samples, self._runs)
            data[idx, :] = StandardScaler().fit_transform(data[idx, :])
        return data[mask]

    @staticmethod
    def _prep_runs(runs: int, blocks: int) -> np.ndarray:
        return np.tile(np.arange(runs), blocks)

    def _load_all_programs(
        self,
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def _get_fname(self, analysis: str, subject: str = "") -> Path:
        dim = getattr(self, "_code_model_dim")
        if subject != "":
            subject = f"_sub{subject}".replace(".mat", "")
        if dim != "":
            dim = f"_dim{dim}"
        fname = self._base_path.joinpath(
            ".cache",
            "representations",
            analysis,
            f"rep_{self._feature}_{self._target}{subject}{dim}.pkl",
        )
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
        return fname

    @abstractmethod
    def _prep_data(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Handled by subclass.")

    def _get_loader(self, subject: Path) -> partial:
        return partial(self._prep_data, subject)

    @lru_cache(maxsize=None)
    def get_data(
        self,
        analysis: str,
        subject: Path = Path(""),
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fname = self._get_fname(analysis, subject.name)
        if fname.exists() and not self._debug:
            with open(fname, "rb") as f:
                data = pkl.load(f)
            return data["X"], data["y"], data["runs"]
        load_data = self._get_loader(subject)
        X, Y, runs = load_data()
        if not self._debug:
            with open(fname, "wb") as f:
                pkl.dump({"X": X, "y": Y, "runs": runs}, f)
        self._logger.info(f"Caching '{fname.name}'.")
        return X, Y, runs


class DataLoaderPRDA(DataLoader):
    def _prep_data(self, k: int = 5):  # type: ignore
        if "+" in self._feature or "+" in self._target:
            raise RuntimeError("PRDA does not support joint variables.")
        programs, content, lang, structure, fnames = self._load_all_programs()
        if self._target in ["task-content", "task-structure"]:
            Y = locals()[self._target.split("-")[1]]
        else:
            Y = ProgramBenchmark(self._target, self._base_path, fnames).fit_transform(
                programs
            )
        X = ProgramEmbedder(self._feature, self._base_path, "").fit_transform(programs)
        runs = self._prep_runs(k, (Y.size // k + 1))[: Y.size]  # kfold CV
        return X, Y, runs

    def _get_loader(self, subject: Path) -> partial:
        return partial(self._prep_data)


class DataLoaderPREA(DataLoaderPRDA):
    def _prep_data(self, k: int = 5):  # type: ignore
        if "+" in self._feature:
            raise RuntimeError("MREA does not support joint variables.")
        if "+" in self._target:
            temp = getattr(self, "_target")
            parts = temp.split("+")
            X = []
            for part in parts:
                setattr(self, "_target", part)
                y, x, runs = super()._prep_data(k)
                if x.ndim == 1:
                    x = OneHotEncoder(sparse=False).fit_transform(x.reshape(-1, 1))
                X.append(x)
            setattr(self, "_target", temp)
            X = np.concatenate(X, axis=1)
            Y = y  # type: ignore
            return X, y, runs
        Y, X, runs = super()._prep_data(k)
        if X.ndim == 1:  # type: ignore
            X = OneHotEncoder(sparse=False).fit_transform(X.reshape(-1, 1))  # type: ignore
        return X, Y, runs


class DataLoaderMVPA(DataLoader):
    def _prep_xyr(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data, parc, content, lang, structure, ident = self._load_brain_data(subject)
        Y, mask = self._prep_code_reps(content, lang, structure, ident)
        X = self._prep_brain_reps(data, parc, mask)
        runs = self._prep_runs(self._runs, self._blocks)[mask]
        return X, Y, runs

    def _prep_xyr_joint(self, subject: Path, comp: str):
        temp = getattr(self, f"_{comp}")
        parts = temp.split("+")
        if comp == "feature":
            X = []
        elif comp == "target":
            Y = []
        else:
            raise RuntimeError("Unsupported component.")
        for part in parts:
            setattr(self, f"_{comp}", part)
            x, y, runs = self._prep_xyr(subject)
            if comp == "feature":
                X.append(x)
            else:
                if y.ndim == 1:
                    y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
                Y.append(y)
        setattr(self, f"_{comp}", temp)
        if comp == "feature":
            X = np.concatenate(X, axis=1)
            Y = y  # type: ignore
        else:
            X = x  # type: ignore
            Y = np.concatenate(Y, axis=1)
        return X, Y, runs

    def _prep_data(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_feature = "+" in self._feature
        joint_target = "+" in self._target
        if joint_feature and joint_target:
            raise RuntimeError("Should only be using one set of joint variables.")
        if joint_feature:
            if "MVPA" not in self._name:
                raise RuntimeError("Only MVPA supports joint features.")
            return self._prep_xyr_joint(subject, "feature")
        if joint_target:
            if "EA" not in self._name:
                raise RuntimeError("Only encoding analyses support joint targets.")
            return self._prep_xyr_joint(subject, "target")
        return self._prep_xyr(subject)


class DataLoaderRSA(DataLoaderMVPA):
    def _prep_data(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Y, runs = super()._prep_data(subject)
        if Y.ndim == 1:
            Y = OneHotEncoder(sparse=False).fit_transform(Y.reshape(-1, 1))
        return X, Y, runs


class DataLoaderVWEA(DataLoaderRSA):
    def _prep_data(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Y, X, runs = super()._prep_data(subject)
        return X, Y, runs


class DataLoaderNLEA(DataLoaderVWEA):
    def _prep_data(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Y, runs = super()._prep_data(subject)
        Y = Y.mean(axis=1).reshape(-1, 1)
        return X, Y, runs


class DataLoaderCKA(DataLoaderRSA):
    pass


class DataLoaderCVWEA(DataLoaderVWEA):
    pass


class DataLoaderCNLEA(DataLoaderNLEA):
    pass
