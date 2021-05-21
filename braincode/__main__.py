import itertools
import warnings
from argparse import ArgumentParser

from joblib import Parallel, delayed
from mvpa import MVPA
from rsa import RSA


def rsa_analysis(network):
    RSA(network).run()


def mvpa_analysis(network, feature):
    MVPA(network, feature).run()


if __name__ == "__main__":
    default = "all"
    analyses = ["rsa", "mvpa"]
    networks = ["lang", "MD", "aud", "vis"]
    features = ["code", "content", "structure", "bow"]
    parser = ArgumentParser(description="run specified analysis type")
    parser.add_argument("analysis", choices=analyses)
    parser.add_argument(
        "-n", "--network", choices=[default] + networks, default=default
    )
    parser.add_argument(
        "-f", "--feature", choices=[default] + features, default=default
    )
    args = parser.parse_args()
    if args.network != default:
        networks = [args.network]
    if args.feature != default:
        features = [args.feature]
    if args.analysis == analyses[0]:
        if args.feature != default:
            warnings.warn("rsa does not use feature; ignoring argument")
        params = [[network] for network in networks]
        function = rsa_analysis
    elif args.analysis == analyses[1]:
        params = list(itertools.product(networks, features))
        function = mvpa_analysis
    else:
        raise argparse.ArgumentTypeError
    Parallel(n_jobs=len(params))(
        delayed(function)(*params[i]) for i in range(len(params))
    )
