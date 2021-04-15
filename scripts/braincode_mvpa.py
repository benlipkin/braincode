import os
import sys
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from braincode_util import *


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
                    if feature == "sent v sent":
                        idx = code == "sent"
                    elif feature == "code v code":
                        idx = code == "code"
                    elif feature == "math v math":
                        idx = (code == "code") & (formatcell(content) == "math")
                    elif feature == "str v str":
                        idx = (code == "code") & (formatcell(content) == "str")
                    elif feature == "seq v seq":
                        idx = (code == "code") & (formatcell(structure) == "seq")
                    elif feature == "for v for":
                        idx = (code == "code") & (formatcell(structure) == "for")
                    elif feature == "if v if":
                        idx = (code == "code") & (formatcell(structure) == "if")
                    else:
                        raise LookupError()
                    n = idx.sum() // 2
                    return np.concatenate((np.zeros(n), np.ones(n))), idx
    return encoder.fit_transform(y[idx]), idx


def get_xy(fname, network, feature):
    data, parc, content, lang, structure = parse_mat(get_mat(fname), network)
    y, idx = prep_y(content, lang, structure, feature)
    X = prep_x(data, parc)[idx]
    return X, y


def init_cmat(n):
    return np.zeros((n, n))


def classifier():
    return LinearSVC(C=1.0, max_iter=1e5)


def crossval(folds=5):
    return KFold(n_splits=folds, shuffle=True, random_state=0)


def accuracy(cmat):
    return np.trace(cmat) / cmat.sum()


def train_and_test_model(X, y, classifier=classifier(), crossval=crossval()):
    classes = np.unique(y)
    cmat = init_cmat(classes.size)
    for train, test in crossval.split(X):
        model = classifier.fit(X[train], y[train])
        cmat += confusion_matrix(y[test], model.predict(X[test]), labels=classes)
    return cmat


def run_decoding_pipeline(input_dir, feature, network, mode, iters=1):
    assert mode in ["test", "null", "diagnostic"]
    if mode in ["null", "diagnostic"]:
        fname = f"../outputs/{mode}_{'_'.join(feature.split())}_{network}.npy"
        if os.path.exists(fname):
            return np.load(fname)
        null = np.zeros((iters))
    for idx in tqdm(range(iters), file=sys.stdout, leave=False):
        cmat = init_cmat(len(feature.split(" v ")))
        for subject in sorted(os.listdir(input_dir)):
            X, y = get_xy(input_dir + subject, network, feature)
            if mode in ["null", "diagnostic"]:
                np.random.shuffle(y)
            cmat += train_and_test_model(X, y)
        if mode == "test":
            return accuracy(cmat)
        null[idx] = accuracy(cmat)
    np.save(fname, null)
    return null


def print_progress(feature, network):
    print(f"\nrunning MVPA pipeline on {feature} in {network}")


def print_results(test, null):
    print(f"acc = {test}\np = {(test < null).sum() / null.size}")


def print_diagnostic(feature, network, diagnostic):
    print(
        f"{feature.split()[0]} {network}:\nmu={diagnostic.mean()}\nsigma={diagnostic.std()}\nmin={diagnostic.min()}\nmax={diagnostic.max()}"
    )


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


def main():
    input_dir = "../inputs/item_data_tvals_20201002/"
    for network in ["lang", "MD", "aud", "vis"]:
        # for feature in ["sent v code", "math v str", "seq v for v if"]:
        #     print_progress(feature, network)
        #     test = run_decoding_pipeline(input_dir, feature, network, "test")
        #     null = run_decoding_pipeline(input_dir, feature, network, "null", iters=1000)
        #     save_results(test, null, feature, network)
        for feature in [
            "sent v sent",
            "code v code",
            "math v math",
            "str v str",
            "seq v seq",
            "for v for",
            "if v if",
        ]:
            print_progress(feature, network)
            diagnostic = run_decoding_pipeline(
                input_dir, feature, network, "diagnostic", iters=100
            )
            print_diagnostic(feature, network, diagnostic)


if __name__ == "__main__":
    main()
