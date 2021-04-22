import itertools
from argparse import ArgumentParser

from joblib import Parallel, delayed
from mvpa import MVPA
from rsa import RSA


def rsa_analysis(network):
    RSA(network).run()


def mvpa_analysis(network, feature):
    MVPA(network, feature).run()


if __name__ == "__main__":
    parser = ArgumentParser(description="run specified analysis type")
    parser.add_argument("analysis", choices=["rsa", "mvpa"])
    args = parser.parse_args()
    networks = ["lang", "MD", "aud", "vis"]
    if args.analysis == "rsa":
        params = [[network] for network in networks]
        function = rsa_analysis
    elif args.analysis == "mvpa":
        features = ["sent v code", "math v str", "seq v for v if"]
        params = list(itertools.product(networks, features))
        function = mvpa_analysis
    else:
        raise argparse.ArgumentTypeError
    Parallel(n_jobs=len(params))(
        delayed(function)(*params[i]) for i in range(len(params))
    )
