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
        self._default = "all"
        self._analyses = [
            "rsa",
            "mvpa",
            "prda",
            "nlea",
        ]
        self._features = [
            "brain-MD+lang",
            "brain-MD+vis",
            "brain-lang+vis",
            "brain-MD",
            "brain-lang",
            "brain-vis",
            "brain-aud",
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
        ]
        self._targets = [
            "test-code",
            "test-lang",
            "task-content",
            "task-structure",
            "task-lines",
            "task-bytes",
            "task-nodes",
            "task-tokens",
            "task-halstead",
            "task-cyclomatic",
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
        ]
        self._logger = logging.getLogger(self.__class__.__name__)

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

    def _prep_analyses(self):
        if not hasattr(self, "_args"):
            raise RuntimeError("CLI args not set. Need to parse first.")
        if self._args.feature != self._default:
            self._features = [self._args.feature]
        if self._args.target != self._default:
            self._targets = [self._args.target]
        if self._args.analysis in ["rsa", "mvpa", "nlea"]:
            self._features = self._clean_arg(self._features, "brain-", "-f")
            if self._args.analysis in ["mvpa", "rsa"]:
                self._targets = self._clean_arg(self._targets, "+", "-t", keep=False)
            if self._args.analysis in ["nlea", "rsa"]:
                self._features = self._clean_arg(self._features, "+", "-f", keep=False)
            if self._args.analysis == "rsa":
                self._targets = self._clean_arg(self._targets, "code-", "-t")
        elif self._args.analysis == "prda":
            self._features = self._clean_arg(self._features, "code-", "-f")
            self._targets = self._clean_arg(self._targets, "task-", "-t")
        else:
            raise ValueError("Invalid argument for analysis.")
        self._kwargs = {
            "base_path": self._args.base_path,
            "score_only": self._args.score_only,
            "code_model_dim": self._args.code_model_dim,
        }
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
