import os
import sys
import itertools
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import LinearSVC
from braincode_util import *


def print_progress(feature, network):
    print(f"\nrunning MVPA pipeline on {feature} in {network}")


def prep_y(content, lang, structure, feature, encoder=LabelEncoder()):
    code = np.array(["sent" if i == "sent" else "code" for i in formatcell(lang)])
    if feature == "sent v code":
        y = code
        idx = np.ones(code.size, dtype="bool")
    else:
        idx = code == "code"
        if feature == "math v str":
            y = formatcell(content)
        else:
            y = formatcell(structure)
            if feature != "seq v for v if":
                if feature == "seq v for":
                    idx = idx & ((y == "seq") | (y == "for"))
                elif feature == "seq v if":
                    idx = idx & ((y == "seq") | (y == "if"))
                elif feature == "for v if":
                    idx = idx & ((y == "for") | (y == "if"))
                else:
                    raise LookupError()
    return encoder.fit_transform(y[idx]), idx


def get_runs():
    return np.tile(np.arange(12), 6)


def get_xyr(fname, network, feature):
    data, parc, content, lang, structure = parse_mat(get_mat(fname), network)
    y, idx = prep_y(content, lang, structure, feature)
    X = prep_x(data, parc)[idx]
    runs = get_runs()[idx]
    return X, y, runs


def init_cmat(n):
    return np.zeros((n, n))


def shuffle_within_runs(y, runs):
    for run in np.unique(runs):
        y[runs == run] = np.random.permutation(y[runs == run])
    return y


def train_and_test_model(X, y, runs):
    classes = np.unique(y)
    classifier = LinearSVC(C=1.0, max_iter=1e5)
    cmat = init_cmat(classes.size)
    for train, test in LeaveOneGroupOut().split(X, y, runs):
        model = classifier.fit(X[train], y[train])
        cmat += confusion_matrix(y[test], model.predict(X[test]), labels=classes)
    return cmat


def accuracy(cmat):
    return np.trace(cmat) / cmat.sum()


def run_decoding_pipeline(feature, network, mode, iters=1):
    assert mode in ["test", "null"]
    if mode == "null":
        fname = f"../outputs/{mode}_{'_'.join(feature.split())}_{network}.npy"
        if os.path.exists(fname):
            return np.load(fname)
        null = np.zeros((iters))
    input_dir = "../inputs/item_data_tvals_20201002/"
    for idx in tqdm(range(iters), leave=False):
        print(f"iter {idx} of {iters}")
        cmat = init_cmat(len(feature.split(" v ")))
        for subject in sorted(os.listdir(input_dir)):
            X, y, runs = get_xyr(input_dir + subject, network, feature)
            if mode == "null":
                y = shuffle_within_runs(y, runs)
            cmat += train_and_test_model(X, y, runs)
        if mode == "test":
            return accuracy(cmat)
        null[idx] = accuracy(cmat)
    np.save(fname, null)
    return null


def print_results(test, null):
    print(f"acc = {test}\np = {(test < null).sum() / null.size}")


def get_range(feature):
    if feature == "sent v code":
        return [0.4, 0.9]
    elif feature == "math v str":
        return [0.4, 0.65]
    elif feature == "seq v for v if":
        return [0.25, 0.55]
    else:
        raise LookupError()


def save_results(test, null, feature, network):
    print_results(test, null)
    plt.hist(null, bins=25, color="lightblue", edgecolor="black")
    plt.axvline(test, color="black", linewidth=3)
    plt.xlim(get_range(feature))
    plt.savefig(f"../plots/hist/{'_'.join(feature.split())}_{network}.png")
    plt.clf()


def main(args):
    feature, network = args
    print_progress(feature, network)
    test = run_decoding_pipeline(feature, network, mode="test")
    null = run_decoding_pipeline(feature, network, mode="null", iters=1000)
    save_results(test, null, feature, network)


if __name__ == "__main__":
    features = ["sent v code", "math v str", "seq v for v if"]
    networks = ["lang", "MD", "aud", "vis"]
    args = list(itertools.product(features, networks))
    Parallel(n_jobs=len(args))(delayed(main)(args[i]) for i in range(len(args)))
