import itertools
import multiprocessing
import warnings
from argparse import ArgumentParser

from decoding import MVPA, PRDA
from joblib import Parallel, delayed
from rsa import RSA


class CLI:
    def __init__(self):
        self._default = "all"
        self._analyses = [
            "rsa",
            "mvpa",
            "prda",
        ]
        self._features = [
            "brain-lang",
            "brain-MD",
            "brain-aud",
            "brain-vis",
            "code-bow",
            "code-tfidf",
            "code-codeberta",
        ]
        self._targets = [
            "test-code",
            "task-content",
            "task-structure",
            "code-bow",
            "code-tfidf",
            "code-codeberta",
        ]

    def _build_parser(self):
        self._parser = ArgumentParser(description="run specified analysis type")
        self._parser.add_argument("analysis", choices=self._analyses)
        self._parser.add_argument(
            "-f",
            "--feature",
            choices=[self._default] + self._features,
            default=self._default,
        )
        self._parser.add_argument(
            "-t",
            "--target",
            choices=[self._default] + self._targets,
            default=self._default,
        )

    def _parse_args(self):
        if not hasattr(self, "_parser"):
            raise RuntimeError("CLI parser not set. Need to build first.")
        self._args = self._parser.parse_args()

    def _clean_arg(self, arg, match, input):
        arg = [opt for opt in arg if match in opt]
        if len(arg) > 0:
            return arg
        else:
            raise ValueError(
                f"{self._args.analysis.upper()} only accepts '{match}' arguments for '{input}'."
            )

    def _prep_analyses(self):
        if not hasattr(self, "_args"):
            raise RuntimeError("CLI args not set. Need to parse first.")
        if self._args.feature != self._default:
            self._features = [self._args.feature]
        if self._args.target != self._default:
            self._targets = [self._args.target]
        if self._args.analysis == "rsa":
            if self._args.target != self._default:
                warnings.warn("rsa does not use target; ignoring argument")
            self._features = self._clean_arg(self._features, "brain", "-f")
            self._params = [[feature] for feature in self._features]
        else:
            if self._args.analysis == "mvpa":
                self._features = self._clean_arg(self._features, "brain", "-f")
            elif self._args.analysis == "prda":
                self._features = self._clean_arg(self._features, "code", "-f")
                self._targets = self._clean_arg(self._targets, "task", "-t")
            else:
                raise ValueError("Invalid argument for analysis.")
            self._params = list(itertools.product(self._features, self._targets))
        self._analysis = globals()[self._args.analysis.upper()]

    def _run_analysis(self, param):
        if not hasattr(self, "_analysis"):
            raise RuntimeError("Analysis type not set. Need to prep first.")
        self._analysis(*param).run()

    def _run_parallel_analyses(self):
        if not hasattr(self, "_params"):
            raise RuntimeError("Analysis parameters not set. Need to prep first.")
        Parallel(n_jobs=min(multiprocessing.cpu_count(), len(self._params)))(
            delayed(self._run_analysis)(param) for param in self._params
        )

    def run_main(self):
        self._build_parser()
        self._parse_args()
        self._prep_analyses()
        self._run_parallel_analyses()
