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


if __name__ == "__main__":
    default = "all"
    analyses = ["rsa", "mvpa"]
    embeddings = ["brain-lang", "brain-MD", "brain-aud", "brain-vis"]
    features = ["task-code", "task-content", "task-structure", "code-bow", "code-tfidf"]
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
        params = [[embedding] for embedding in embeddings]
        function = rsa_analysis
    elif args.analysis == analyses[1]:
        params = list(itertools.product(embeddings, features))
        function = mvpa_analysis
    else:
        raise argparse.ArgumentError()
    Parallel(n_jobs=len(params))(delayed(function)(*param) for param in params)
