import itertools
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
        self._embeddings = [
            "brain-lang",
            "brain-MD",
            "brain-aud",
            "brain-vis",
            "code-bow",
            "code-tfidf",
            # "code-seq2seq",
        ]
        self._features = [
            "test-code",
            "task-content",
            "task-structure",
            "code-bow",
            "code-tfidf",
            # "code-seq2seq",
        ]

    def _build_parser(self):
        self._parser = ArgumentParser(description="run specified analysis type")
        self._parser.add_argument("analysis", choices=self._analyses)
        self._parser.add_argument(
            "-e",
            "--embedding",
            choices=[self._default] + self._embeddings,
            default=self._default,
        )
        self._parser.add_argument(
            "-f",
            "--feature",
            choices=[self._default] + self._features,
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
        if self._args.embedding != self._default:
            self._embeddings = [self._args.embedding]
        if self._args.feature != self._default:
            self._features = [self._args.feature]
        if self._args.analysis == "rsa":
            if self._args.feature != self._default:
                warnings.warn("rsa does not use feature; ignoring argument")
            self._embeddings = self._clean_arg(self._embeddings, "brain", "-e")
            self._params = [[embedding] for embedding in self._embeddings]
        else:
            if self._args.analysis == "mvpa":
                self._embeddings = self._clean_arg(self._embeddings, "brain", "-e")
            elif self._args.analysis == "prda":
                self._embeddings = self._clean_arg(self._embeddings, "code", "-e")
                self._features = self._clean_arg(self._features, "task", "-f")
            else:
                raise ValueError("Invalid argument for analysis.")
            self._params = list(itertools.product(self._embeddings, self._features))
        self._analysis = globals()[self._args.analysis.upper()]

    def _run_analysis(self, param):
        if not hasattr(self, "_analysis"):
            raise RuntimeError("Analysis type not set. Need to prep first.")
        self._analysis(*param).run()

    def _run_parallel_analyses(self):
        if not hasattr(self, "_params"):
            raise RuntimeError("Analysis parameters not set. Need to prep first.")
        Parallel(n_jobs=len(self._params))(
            delayed(self._run_analysis)(param) for param in self._params
        )

    def run_main(self):
        self._build_parser()
        self._parse_args()
        self._prep_analyses()
        self._run_parallel_analyses()


if __name__ == "__main__":
    CLI().run_main()
