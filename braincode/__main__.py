import itertools
import warnings
from argparse import ArgumentParser

from joblib import Parallel, delayed
from mvpa import MVPA
from rsa import RSA


def rsa_analysis(embedding):
    RSA(embedding).run()


def mvpa_analysis(embedding, feature):
    MVPA(embedding, feature).run()


def prda_analysis(embedding, feature):
    PRDA(embedding, feature).run()


def clean_args(args, match, analysis, input):
    args = [arg for arg in args if match in arg]
    if len(args) == 0:
        raise ValueError(
            f"{analysis.upper()} only accepts '{match}' arguments for '{input}'."
        )
    return args


if __name__ == "__main__":
    default = "all"
    analyses = ["rsa", "mvpa", "prda"]
    embeddings = [
        "brain-lang",
        "brain-MD",
        "brain-aud",
        "brain-vis",
        "code-bow",
        "code-tfidf",
        # "code-seq2seq",
    ]
    features = [
        "task-code",
        "task-content",
        "task-structure",
        "code-bow",
        "code-tfidf",
        # "code-seq2seq",
    ]
    parser = ArgumentParser(description="run specified analysis type")
    parser.add_argument("analysis", choices=analyses)
    parser.add_argument(
        "-e", "--embedding", choices=[default] + embeddings, default=default
    )
    parser.add_argument(
        "-f", "--feature", choices=[default] + features, default=default
    )
    args = parser.parse_args()
    if args.embedding != default:
        embeddings = [args.embedding]
    if args.feature != default:
        features = [args.feature]
    if args.analysis == analyses[0]:
        if args.feature != default:
            warnings.warn("rsa does not use feature; ignoring argument")
        embeddings = clean_args(embeddings, "brain", args.analysis, "-e")
        params = [[embedding] for embedding in embeddings]
        function = rsa_analysis
    elif args.analysis == analyses[1]:
        embeddings = clean_args(embeddings, "brain", args.analysis, "-e")
        params = list(itertools.product(embeddings, features))
        function = mvpa_analysis
    elif args.analysis == analyses[2]:
        embeddings = clean_args(embeddings, "code", args.analysis, "-e")
        features = clean_args(features, "task", args.analysis, "-f")
        params = list(itertools.product(embeddings, features))
        function = prda_analysis
    else:
        raise ValueError("Invalid Argument")
    Parallel(n_jobs=len(params))(delayed(function)(*param) for param in params)
