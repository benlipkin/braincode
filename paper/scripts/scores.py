import itertools

import numpy as np
import pandas as pd
import scipy.stats as st


def make_table(name, analysis, features, targets):
    pairs = list(itertools.product(features, targets))
    table = pd.DataFrame(pairs, columns=["Feature", "Target"])
    scores = []
    subjects = []
    null_mu = []
    null_std = []
    for i, row in table.iterrows():
        scores.append(
            np.load(
                f"../../braincode/.cache/scores/{name}/score_{row.Feature}_{row.Target}.npy"
            )
        )
        if name != "prda":
            subjects.append(
                np.load(
                    f"../../braincode/.cache/scores/{name}/subjects_{row.Feature}_{row.Target}.npy"
                )
            )
        null = np.load(
            f"../../braincode/.cache/scores/{name}/null_{row.Feature}_{row.Target}.npy"
        )
        null_mu.append(null.mean())
        null_std.append(null.std())
    table["Score"] = np.array(scores)
    if name != "prda":
        table["95CI"] = 1.96 * st.sem(np.array(subjects), axis=1)
    table["Null Mean"] = np.array(null_mu)
    table["Null SD"] = np.array(null_std)
    table["z"] = (table["Score"] - table["Null Mean"]) / table["Null SD"]
    table["p (corrected)"] = st.norm.sf(table["z"]) * table.shape[0]
    table["h (corrected)"] = (table["p (corrected)"] < 0.01).astype(int)
    table.to_csv(f"../tables/raw/{analysis}.csv", index=False)


def make_subjects_table(name, analysis, features, targets):
    pairs = list(itertools.product(features, targets))
    table = pd.DataFrame(pairs, columns=["Feature", "Target"])
    scores = []
    for i, row in table.iterrows():
        scores.append(
            np.load(
                f"../../braincode/.cache/scores/{name}/subjects_{row.Feature}_{row.Target}.npy"
            )
        )
    table = pd.concat((table, pd.DataFrame(scores)), axis=1)
    table.columns = [
        col if isinstance(col, str) else f"Subject_{col+1}" for col in table.columns
    ]
    table.to_csv(f"../tables/raw/{analysis}_subjects.csv", index=False)


def make_table0():
    name = "prda"
    analysis = "prda_properties"
    features = [
        "random",
        "codeberta",
        "ct",
        "xlnet",
        "seq2seq",
        "bow",
        "tfidf",
    ]
    targets = [
        "content",
        "structure",
        "tokens",
        "lines",
    ]
    make_table(name, analysis, features, targets)


def make_table1():
    name = "mvpa"
    analysis = "mvpa_properties_cls"
    features = [
        "MD",
        "lang",
        "vis",
        "aud",
    ]
    targets = [
        "code",
        "lang",
        "content",
        "structure",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table2():
    name = "mvpa"
    analysis = "mvpa_properties_rgr"
    features = [
        "MD",
        "lang",
        "vis",
        "aud",
    ]
    targets = [
        "tokens",
        "lines",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table3():
    name = "mvpa"
    analysis = "mvpa_models"
    features = [
        "MD",
        "lang",
        "vis",
        "aud",
    ]
    targets = [
        "random",
        "codeberta",
        "ct",
        "xlnet",
        "seq2seq",
        "bow",
        "tfidf",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table4():
    name = "mvpa"
    analysis = "mvpa_properties_cls_ablation"
    features = [
        "MD+lang+vis+aud",
        "MD+lang+vis",
        "MD+lang",
        "MD",
    ]
    targets = [
        "code",
        "lang",
        "content",
        "structure",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table5():
    name = "mvpa"
    analysis = "mvpa_properties_rgr_ablation"
    features = [
        "MD+lang+vis+aud",
        "MD+lang+vis",
        "MD+lang",
        "MD",
    ]
    targets = [
        "tokens",
        "lines",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table6():
    name = "mvpa"
    analysis = "mvpa_models_ablation"
    features = [
        "MD+lang+vis+aud",
        "MD+lang+vis",
        "MD+lang",
        "MD",
    ]
    targets = [
        "random",
        "codeberta",
        "ct",
        "xlnet",
        "seq2seq",
        "bow",
        "tfidf",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table7():
    name = "mvpa"
    analysis = "mvpa_properties_all"
    features = [
        "MD",
        "lang",
        "vis",
        "aud",
    ]
    targets = [
        "code",
        "lang",
        "content",
        "structure",
        "tokens",
        "lines",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table8():
    name = "mvpa"
    analysis = "mvpa_properties_all_ablation"
    features = [
        "MD+lang+vis+aud",
        "MD+lang+vis",
        "MD+lang",
        "MD",
    ]
    targets = [
        "code",
        "lang",
        "content",
        "structure",
        "tokens",
        "lines",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table9():
    name = "mvpa"
    analysis = "mvpa_properties_supplemental"
    features = [
        "MD",
        "lang",
        "vis",
        "aud",
    ]
    targets = [
        "tokens",
        "nodes",
        "halstead",
        "cyclomatic",
        "lines",
        "bytes",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table10():
    name = "mvpa"
    analysis = "mvpa_properties_supplemental_ablation"
    features = [
        "MD+lang+vis+aud",
        "MD+lang+vis",
        "MD+lang",
        "MD",
    ]
    targets = [
        "tokens",
        "nodes",
        "halstead",
        "cyclomatic",
        "lines",
        "bytes",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


if __name__ == "__main__":
    make_table0()
    make_table1()
    make_table2()
    make_table3()
    make_table4()
    make_table5()
    make_table6()
    make_table7()
    make_table8()
    make_table9()
    make_table10()
