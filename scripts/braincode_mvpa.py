import os
import numpy as np
from scipy.stats import binom_test
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
            if feature == "seq v for":
                idx = idx & ((y == "seq") | (y == "for"))
            elif feature == "seq v if":
                idx = idx & ((y == "seq") | (y == "if"))
            elif feature == "for v if":
                idx = idx & ((y == "for") | (y == "if"))
            else:
                raise LookupError()
    return encoder.fit_transform(y[idx]), idx


def get_xy(fname, network, feature):
    data, parc, content, lang, structure = parse_mat(get_mat(fname), network)
    y, idx = prep_y(content, lang, structure, feature)
    X = prep_x(data, parc)[idx]
    return X, y


def classifier():
    return LinearSVC(C=1.0, max_iter=10000)


def crossval(folds=5):
    return KFold(n_splits=folds, shuffle=True, random_state=0)


def train_and_test_model(X, y, classifier=classifier()):
    cmat = np.zeros((np.unique(y).size, np.unique(y).size))
    for train, test in crossval().split(X):
        model = classifier.fit(X[train], y[train])
        cmat += confusion_matrix(y[test], model.predict(X[test]))
    return cmat


def run_decoding_pipeline(fname, network, feature):
    X, y = get_xy(fname, network, feature)
    return train_and_test_model(X, y)


def k(cmat):
    return np.trace(cmat)


def n(cmat):
    return cmat.sum()


def p(cmat):
    return cmat.sum(axis=1).max() / n(cmat)


def accuracy(cmat):
    return k(cmat) / n(cmat)


def stats(cmat):
    return binom_test(k(cmat), n(cmat), p(cmat), alternative="greater")


def summarize(network, feature, subject, cmat):
    return "%s,%s,%s,%d,%f,%d,%f,%f\n" % (
        network,
        feature,
        subject,
        n(cmat),
        p(cmat),
        k(cmat),
        accuracy(cmat),
        stats(cmat),
    )


def main():
    ostream = "NETWORK,FEATURE,SUBJECT,N,P,K,ACC,PVAL\n"
    input_dir = "../inputs/item_data_tvals_20201002/"
    networks = ["lang", "MD", "aud"]
    features = ["sent v code", "math v str", "seq v for", "seq v if", "for v if"]
    for network in networks:
        for feature in features:
            for subject in sorted(os.listdir(input_dir)):
                ostream += summarize(
                    network,
                    feature,
                    subject,
                    run_decoding_pipeline(input_dir + subject, network, feature),
                )
    with open("../outputs/results.csv", "w") as f:
        f.write(ostream)


if __name__ == "__main__":
    main()
