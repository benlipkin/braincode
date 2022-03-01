import itertools
import logging
import multiprocessing
import sys
from argparse import ArgumentParser
from pathlib import Path

from decoding import MVPA, PRDA
from encoding import NLEA
from joblib import Parallel, delayed, parallel_backend
from rsa import RSA


class CLI:
    def __init__(self):
        self._default_path = Path(__file__).parent
        self._default_arg = "all"
        self._analyses = ["rsa", "mvpa", "prda", "nlea"]
        self._features = (
            self._brain_networks + self._code_models + self._expanded_features
        )
        self._targets = (
            self._code_benchmarks + self._code_models
        )  # + self._expanded_targets
        self._logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _base_args(prefix, units):
        return [f"{prefix}-{i}" for i in units]

    @staticmethod
    def _joint_args(prefix, units):
        return [f"{prefix}-{i}+{j}" for i, j in list(itertools.combinations(units, 2))]

    @property
    def _brain_networks(self):
        prefix = "brain"
        units = ["MD", "lang", "vis", "aud"]
        return self._base_args(prefix, units)

    @property
    def _code_models(self):
        prefix = "code"
        base_models = ["projection", "bow", "tfidf", "seq2seq"]
        transformers = ["xlnet", "bert", "gpt2", "transformer", "roberta"]
        units = base_models + transformers
        return self._base_args(prefix, units)

    @property
    def _code_benchmarks(self):
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
    def _expanded_features(self):
        prefix = "brain"
        units = ["MD", "lang", "vis"]
        return self._joint_args(prefix, units)

    @property
    def _expanded_targets(self):
        prefix = "task"
        units = ["content", "structure", "tokens", "lines"]
        return self._joint_args(prefix, units)

    def _build_parser(self):
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
        self._parser.add_argument("-s", "--score_only", action="store_true")
        self._parser.add_argument("-d", "--code_model_dim", default="")
        self._parser.add_argument("-p", "--base_path", default=self._default_path)

    def _parse_args(self):
        if not hasattr(self, "_parser"):
            raise RuntimeError("CLI parser not set. Need to build first.")
        self._args = self._parser.parse_args()

    def _clean_arg(self, arg, match, input, keep=True):
        arg = [opt for opt in arg if ((match in opt) == keep)]
        if len(arg) > 0:
            return arg
        else:
            raise ValueError(
                f"{self._args.analysis.upper()} only accepts '{match}' arguments for '{input}'."
            )

    def _prep_args(self):
        if self._args.feature != self._default_arg:
            self._features = [self._args.feature]
        if self._args.target != self._default_arg:
            self._targets = [self._args.target]
        if self._args.analysis in ["rsa", "mvpa", "nlea"]:
            self._features = self._clean_arg(self._features, "brain-", "-f")
        if self._args.analysis in ["rsa", "mvpa", "prda"]:
            self._targets = self._clean_arg(self._targets, "+", "-t", keep=False)
        if self._args.analysis in ["rsa", "nlea"]:
            self._features = self._clean_arg(self._features, "+", "-f", keep=False)
        if self._args.analysis == "rsa":
            self._targets = self._clean_arg(self._targets, "code-", "-t")
        if self._args.analysis == "prda":
            self._features = self._clean_arg(self._features, "code-", "-f")
            self._targets = self._clean_arg(self._targets, "task-", "-t")

    def _prep_kwargs(self):
        self._kwargs = {
            "base_path": self._args.base_path,
            "score_only": self._args.score_only,
            "code_model_dim": self._args.code_model_dim,
        }

    def _prep_analyses(self):
        if not hasattr(self, "_args"):
            raise RuntimeError("CLI args not set. Need to parse first.")
        self._prep_args()
        self._prep_kwargs()
        self._params = list(
            itertools.product(self._features, self._targets, [self._kwargs])
        )
        self._analysis = globals()[self._args.analysis.upper()]

    def _run_analysis(self, param):
        if not hasattr(self, "_analysis"):
            raise RuntimeError("Analysis type not set. Need to prep first.")
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self._analysis(*param).run()

    def _run_parallel_analyses(self):
        if not hasattr(self, "_params"):
            raise RuntimeError("Analysis parameters not set. Need to prep first.")
        n_jobs = min(multiprocessing.cpu_count(), len(self._params))
        self._logger.info(
            f"Running {self._analysis.__name__} for each set of {len(self._params)} analysis parameters using {n_jobs} CPUs."
        )
        with parallel_backend("loky", n_jobs=n_jobs):
            Parallel()(delayed(self._run_analysis)(param) for param in self._params)

    def run_main(self):
        self._build_parser()
        self._parse_args()
        self._prep_analyses()
        self._run_parallel_analyses()
