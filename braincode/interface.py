import itertools
import logging
import multiprocessing
import re
import sys
import typing
from argparse import ArgumentParser
from pathlib import Path

from joblib import Parallel, delayed, parallel_backend

import braincode
from braincode.abstract import Object
from braincode.analyses import Analysis


class CLI(Object):
    def __init__(self) -> None:
        super().__init__()
        self._default_path = Path(__file__).parent
        self._default_arg = "all"
        self._analyses = [
            "mvpa",
            "rsa",
            "vwea",
            "nlea",
            "cvwea",
            "cnlea",
            "prda",
            "prea",
        ]
        self._features = self._brain_networks + self._code_models
        self._targets = self._code_benchmarks + self._code_models

    @staticmethod
    def _base_args(prefix: str, units: typing.List[str]) -> typing.List[str]:
        return [f"{prefix}-{i}" for i in units]

    @staticmethod
    def _joint_args(prefix: str, units: typing.List[str]) -> typing.List[str]:
        return [
            f"{prefix}-{i}+{prefix}-{j}" for i, j in itertools.combinations(units, 2)
        ]

    @staticmethod
    def _max_arg(prefix: str, units: typing.List[str]) -> typing.List[str]:
        arg = ""
        for unit in units:
            arg += f"{prefix}-{unit}+"
        return [arg.strip("+")]

    @property
    def _brain_networks(self) -> typing.List[str]:
        prefix = "brain"
        units = ["md_lh", "md_rh", "lang_lh", "lang_rh"]
        return self._base_args(prefix, units)

    @property
    def _code_models(self) -> typing.List[str]:
        prefix = "code"
        units = [
            "tokens",
            "llm_350m_nl",
            "llm_350m_mono",
            "llm_2b_nl",
            "llm_2b_mono",
            "llm_6b_nl",
            "llm_6b_mono",
            "llm_16b_nl",
            "llm_16b_mono",
        ]
        return self._base_args(prefix, units)

    @property
    def _code_benchmarks(self) -> typing.List[str]:
        prefix = "task"
        units = ["content", "structure", "nodes", "lines"]
        return ["test-code"] + self._base_args(prefix, units)

    def _build_parser(self) -> None:
        self._parser = ArgumentParser(description="run specified analysis type")
        self._parser.add_argument("analysis", choices=self._analyses)
        self._parser.add_argument("-f", "--feature", default=self._default_arg)
        self._parser.add_argument("-t", "--target", default=self._default_arg)
        self._parser.add_argument("-m", "--metric", default="")
        self._parser.add_argument("-d", "--code_model_dim", default="")
        self._parser.add_argument("-p", "--base_path", default=self._default_path)
        self._parser.add_argument("-s", "--score_only", action="store_true")
        self._parser.add_argument("-b", "--debug", action="store_true")

    def _parse_args(self) -> None:
        if not hasattr(self, "_parser"):
            raise RuntimeError("CLI parser not set. Need to build first.")
        self._args = self._parser.parse_args()

    def _clean_arg(self, arg, match, flag, keep=True) -> typing.List[str]:
        arg = [opt for opt in arg if (match in opt) == keep]
        if len(arg) > 0:
            return arg
        tag = "only accepts" if keep else "does not accept"
        raise ValueError(
            f"{self._args.analysis.upper()} {tag} '{match}' arguments for '{flag}'."
        )

    def _prep_args(self) -> None:
        if "*" in self._args.feature:
            r = re.compile(self._args.feature)
            self._features = list(filter(r.match, self._features))
        elif self._args.feature != self._default_arg:
            self._features = [self._args.feature]
        if "*" in self._args.target:
            r = re.compile(self._args.target)
            self._targets = list(filter(r.match, self._targets))
        elif self._args.target != self._default_arg:
            self._targets = [self._args.target]
        if self._args.analysis not in ["prda", "prea"]:
            self._features = self._clean_arg(self._features, "brain-", "-f")
        if self._args.analysis not in ["vwea", "nlea", "prea"]:
            self._targets = self._clean_arg(self._targets, "+", "-t", keep=False)
        if self._args.analysis not in ["mvpa", "prda"]:
            self._features = self._clean_arg(self._features, "+", "-f", keep=False)
        if self._args.analysis == "rsa":
            self._targets = self._clean_arg(self._targets, "test-", "-t", keep=False)
        if self._args.analysis == "cka":
            self._targets = self._clean_arg(self._targets, "code-", "-t")
        if self._args.analysis in ["prda", "prea"]:
            self._features = self._clean_arg(self._features, "code-", "-f")
            self._targets = self._clean_arg(self._targets, "task-", "-t")
        if self._args.analysis in ["cnlea", "cvwea"]:
            self._targets = ["test-code"]

    def _prep_kwargs(self) -> None:
        self._kwargs = {
            "metric": self._args.metric,
            "code_model_dim": self._args.code_model_dim,
            "base_path": self._args.base_path,
            "score_only": self._args.score_only,
            "debug": self._args.debug,
        }

    def _prep_analyses(self) -> None:
        if not hasattr(self, "_args"):
            raise RuntimeError("CLI args not set. Need to parse first.")
        self._prep_args()
        self._prep_kwargs()
        self._params = list(itertools.product(self._features, self._targets))
        self._analysis = getattr(braincode, self._args.analysis.upper())
        if not issubclass(self._analysis, Analysis):
            raise ValueError("Invalid analysis type.")

    def _run_analysis(self, args: typing.Tuple[str, str], kwargs: dict) -> None:
        if not hasattr(self, "_analysis"):
            raise RuntimeError("Analysis type not set. Need to prep first.")
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self._analysis(*args, **kwargs).run()

    def _run_parallel_analyses(self) -> None:
        if not hasattr(self, "_params"):
            raise RuntimeError("Analysis parameters not set. Need to prep first.")
        n_jobs = min(multiprocessing.cpu_count(), len(self._params))
        self._logger.info(
            f"Running {self._analysis.__name__} "
            + f"for each set of {len(self._params)} analysis configurations "
            + f"using {n_jobs} CPUs."
        )
        with parallel_backend("loky", n_jobs=n_jobs):
            Parallel()(
                delayed(self._run_analysis)(args, self._kwargs) for args in self._params
            )

    def run_main(self) -> None:
        self._build_parser()
        self._parse_args()
        self._prep_analyses()
        self._run_parallel_analyses()
