import itertools
import warnings
from argparse import ArgumentParser

from decoding import MVPA, PRDA
from joblib import Parallel, delayed
from rsa import RSA


def analysis_options():
    return ["rsa", "mvpa", "prda"]


def embedding_options():
    return [
        "brain-lang",
        "brain-MD",
        "brain-aud",
        "brain-vis",
        "code-bow",
        "code-tfidf",
        # "code-seq2seq",
    ]


def feature_options():
    return [
        "task-code",
        "task-content",
        "task-structure",
        "code-bow",
        "code-tfidf",
        # "code-seq2seq",
    ]


def build_parser(default, analyses, embeddings, features):
    parser = ArgumentParser(description="run specified analysis type")
    parser.add_argument("analysis", choices=analyses)
    parser.add_argument(
        "-e", "--embedding", choices=[default] + embeddings, default=default
    )
    parser.add_argument(
        "-f", "--feature", choices=[default] + features, default=default
    )
    return parser


def clean_arg(arg, match, analysis, input):
    arg = [opt for opt in arg if match in opt]
    if len(arg) > 0:
        return arg
    else:
        raise ValueError(
            f"{analysis.upper()} only accepts '{match}' arguments for '{input}'."
        )


def rsa_analysis(embedding):
    RSA(embedding).run()


def mvpa_analysis(embedding, feature):
    MVPA(embedding, feature).run()


def prda_analysis(embedding, feature):
    PRDA(embedding, feature).run()


def prep_analysis(args, default, analyses, embeddings, features):
    if args.embedding != default:
        embeddings = [args.embedding]
    if args.feature != default:
        features = [args.feature]
    if args.analysis == analyses[0]:
        if args.feature != default:
            warnings.warn("rsa does not use feature; ignoring argument")
        embeddings = clean_arg(embeddings, "brain", args.analysis, "-e")
        params = [[embedding] for embedding in embeddings]
        function = rsa_analysis
    elif args.analysis == analyses[1]:
        embeddings = clean_arg(embeddings, "brain", args.analysis, "-e")
        params = list(itertools.product(embeddings, features))
        function = mvpa_analysis
    elif args.analysis == analyses[2]:
        embeddings = clean_arg(embeddings, "code", args.analysis, "-e")
        features = clean_arg(features, "task", args.analysis, "-f")
        params = list(itertools.product(embeddings, features))
        function = prda_analysis
    else:
        raise ValueError("Invalid Argument")
    return params, function


if __name__ == "__main__":
    default = "all"
    analyses = analysis_options()
    embeddings = embedding_options()
    features = feature_options()
    args = build_parser(default, analyses, embeddings, features).parse_args()
    params, function = prep_analysis(args, default, analyses, embeddings, features)
    Parallel(n_jobs=len(params))(delayed(function)(*param) for param in params)
