import itertools
import logging
import multiprocessing
import sys
import typing
from argparse import ArgumentParser
from pathlib import Path

from joblib import Parallel, delayed, parallel_backend

from braincode.decoding import *
from braincode.encoding import *
from braincode.similarity import *


class CLI:
    def __init__(self) -> None:
        self._default_path = Path(__file__).parent
        self._default_arg = "all"
        self._analyses = ["mvpa", "rsa", "vwea", "nlea", "prda"]
        self._features = self._brain_networks + self._code_models + self._joint_networks
        self._targets = self._code_benchmarks + self._code_models + self._max_benchmarks
        self._logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _base_args(prefix: str, units: typing.List[str]) -> typing.List[str]:
        return [f"{prefix}-{i}" for i in units]

    @staticmethod
    def _joint_args(prefix: str, units: typing.List[str]) -> typing.List[str]:
        return [f"{prefix}-{i}+{j}" for i, j in list(itertools.combinations(units, 2))]

    @staticmethod
    def _max_arg(prefix: str, units: typing.List[str]) -> typing.List[str]:
        arg = f"{prefix}-"
        for unit in units:
            arg += f"{unit}+"
        return [arg.strip("+")]

    @property
    def _brain_networks(self) -> typing.List[str]:
        prefix = "brain"
        units = ["MD", "lang", "vis", "aud"]
        return self._base_args(prefix, units)

    @property
    def _code_models(self) -> typing.List[str]:
        prefix = "code"
        base_models = ["projection", "bow", "tfidf", "seq2seq"]
        transformers = ["xlnet", "bert", "gpt2", "transformer", "roberta"]
        units = base_models + transformers
        return self._base_args(prefix, units)

    @property
    def _code_benchmarks(self) -> typing.List[str]:
        prefix = ("test", "task")
        test_tasks = ["code", "lang"]
        base_tasks = ["content", "structure", "tokens", "lines"]
        extra_tasks = ["nodes", "bytes", "halstead", "cyclomatic"]
        units = (test_tasks, base_tasks + extra_tasks)
        return list(
            itertools.chain.from_iterable(
                self._base_args(p, u) for p, u in zip(prefix, units)
            )
        )

    @property
    def _joint_networks(self) -> typing.List[str]:
        prefix = "brain"
        units = ["MD", "lang", "vis"]
        return self._joint_args(prefix, units)

    @property
    def _max_benchmarks(self) -> typing.List[str]:
        prefix = "task"
        units = ["content", "structure", "tokens", "lines"]
        return self._max_arg(prefix, units)

    def _build_parser(self) -> None:
        self._parser = ArgumentParser(description="run specified analysis type")
        self._parser.add_argument("analysis", choices=self._analyses)
        self._parser.add_argument(
            "-f",
            "--feature",
            choices=[self._default_arg] + self._features,
            default=self._default_arg,
        )
        self._parser.add_argument(
            "-t",
            "--target",
            choices=[self._default_arg] + self._targets,
            default=self._default_arg,
        )
        self._parser.add_argument("-m", "--metric", default="")
        self._parser.add_argument("-d", "--code_model_dim", default="")
        self._parser.add_argument("-p", "--base_path", default=self._default_path)
        self._parser.add_argument("-s", "--score_only", action="store_true")
        self._parser.add_argument("-b", "--debug", action="store_true")

    def _parse_args(self) -> None:
        if not hasattr(self, "_parser"):
            raise RuntimeError("CLI parser not set. Need to build first.")
        self._args = self._parser.parse_args()

    def _clean_arg(self, arg, match, input, keep=True) -> typing.List[str]:
        arg = [opt for opt in arg if ((match in opt) == keep)]
        if len(arg) > 0:
            return arg
        else:
            if keep:
                tag = "only accepts"
            else:
                tag = "does not accept"
            raise ValueError(
                f"{self._args.analysis.upper()} {tag} '{match}' arguments for '{input}'."
            )

    def _prep_args(self) -> None:
        if self._args.feature != self._default_arg:
            self._features = [self._args.feature]
        if self._args.target != self._default_arg:
            self._targets = [self._args.target]
        if self._args.analysis != "prda":
            self._features = self._clean_arg(self._features, "brain-", "-f")
        if self._args.analysis not in ["vwea", "nlea"]:
            self._targets = self._clean_arg(self._targets, "+", "-t", keep=False)
        if self._args.analysis not in ["mvpa", "prda"]:
            self._features = self._clean_arg(self._features, "+", "-f", keep=False)
        if self._args.analysis == "rsa":
            self._targets = self._clean_arg(self._targets, "test-", "-t", keep=False)
        if self._args.analysis == "cka":
            self._targets = self._clean_arg(self._targets, "code-", "-t")
        if self._args.analysis == "prda":
            self._features = self._clean_arg(self._features, "code-", "-f")
            self._targets = self._clean_arg(self._targets, "task-", "-t")

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
        self._analysis = globals()[self._args.analysis.upper()]

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
            f"Running {self._analysis.__name__} for each set of {len(self._params)} analysis configurations using {n_jobs} CPUs."
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
