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
    table["h (corrected)"] = (table["p (corrected)"] < 1e-2).astype(int)
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


def make_table_prda_properties():
    name = "prda"
    analysis = "prda_properties"
    features = [
        "projection",
        "roberta",
        "transformer",
        "bert",
        "gpt2",
        "xlnet",
        "seq2seq",
        "tfidf",
        "bow",
    ]
    targets = [
        "content",
        "structure",
        "tokens",
        "lines",
    ]
    make_table(name, analysis, features, targets)


def make_table_mvpa_properties_cls():
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


def make_table_mvpa_properties_rgr():
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


def make_table_mvpa_models():
    name = "mvpa"
    analysis = "mvpa_models"
    features = [
        "MD",
        "lang",
        "vis",
        "aud",
    ]
    targets = [
        "projection",
        "roberta",
        "transformer",
        "bert",
        "gpt2",
        "xlnet",
        "seq2seq",
        "tfidf",
        "bow",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table_mvpa_properties_cls_ablation():
    name = "mvpa"
    analysis = "mvpa_properties_cls_ablation"
    features = [
        "MD+lang",
        "MD+vis",
        "lang+vis",
        "MD",
        "lang",
        "vis",
    ]
    targets = [
        "code",
        "lang",
        "content",
        "structure",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table_mvpa_properties_rgr_ablation():
    name = "mvpa"
    analysis = "mvpa_properties_rgr_ablation"
    features = [
        "MD+lang",
        "MD+vis",
        "lang+vis",
        "MD",
        "lang",
        "vis",
    ]
    targets = [
        "tokens",
        "lines",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table_mvpa_models_ablation():
    name = "mvpa"
    analysis = "mvpa_models_ablation"
    features = [
        "MD+lang",
        "MD+vis",
        "lang+vis",
        "MD",
        "lang",
        "vis",
    ]
    targets = [
        "projection",
        "roberta",
        "transformer",
        "bert",
        "gpt2",
        "xlnet",
        "seq2seq",
        "tfidf",
        "bow",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table_mvpa_properties_all():
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


def make_table_mvpa_properties_all_ablation():
    name = "mvpa"
    analysis = "mvpa_properties_all_ablation"
    features = [
        "MD+lang",
        "MD+vis",
        "lang+vis",
        "MD",
        "lang",
        "vis",
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


def make_table_mvpa_properties_supplemental():
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


def make_table_mvpa_properties_supplemental_ablation():
    name = "mvpa"
    analysis = "mvpa_properties_supplemental_ablation"
    features = [
        "MD+lang",
        "MD+vis",
        "lang+vis",
        "MD",
        "lang",
        "vis",
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


def make_table_rsa_models():
    name = "rsa"
    analysis = "rsa_models"
    features = [
        "MD",
        "lang",
    ]
    targets = [
        "transformer",
        "roberta",
        "gpt2",
        "bert",
        "xlnet",
        "seq2seq",
        "tfidf",
        "bow",
        "projection",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table_rsa_properties():
    name = "rsa"
    analysis = "rsa_properties"
    features = [
        "MD",
        "lang",
    ]
    targets = [
        "content",
        "structure",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table_nlea_all():
    name = "nlea"
    analysis = "nlea_all"
    features = [
        "MD",
        "lang",
    ]
    targets = [
        "code",
        "lang",
        "content",
        "structure",
        "tokens",
        "lines",
        "projection",
        "bow",
        "tfidf",
        "seq2seq",
        "xlnet",
        "bert",
        "gpt2",
        "roberta",
        "transformer",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table_nlea_properties():
    name = "nlea"
    analysis = "nlea_properties"
    features = [
        "MD",
        "lang",
    ]
    targets = [
        "content",
        "structure",
        "tokens",
        "lines",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table_nlea_properties_max():
    name = "nlea"
    analysis = "nlea_properties_max"
    features = [
        "MD",
        "lang",
        "vis",
        "aud",
    ]
    targets = [
        "content+structure+tokens+lines",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table_vwea_models():
    name = "vwea"
    analysis = "vwea_models"
    features = [
        "MD",
        "lang",
    ]
    targets = [
        "transformer",
        "roberta",
        "gpt2",
        "bert",
        "xlnet",
        "seq2seq",
        "tfidf",
        "bow",
        "projection",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_table_mvpa_models_mdls():
    name = "mvpa"
    analysis = "mvpa_models_mdls"
    features = [
        "MD",
        "lang",
    ]
    targets = [
        "transformer",
        "roberta",
        "gpt2",
        "bert",
        "xlnet",
        "seq2seq",
        "tfidf",
        "bow",
        "projection",
    ]
    make_table(name, analysis, features, targets)
    make_subjects_table(name, analysis, features, targets)


def make_core_analyses():
    make_table_mvpa_properties_cls()
    make_table_mvpa_properties_rgr()
    make_table_mvpa_models()
    make_table_mvpa_properties_cls_ablation()
    make_table_mvpa_properties_rgr_ablation()
    make_table_mvpa_models_ablation()
    make_table_mvpa_properties_all()
    make_table_mvpa_properties_all_ablation()
    make_table_mvpa_properties_supplemental()
    make_table_mvpa_properties_supplemental_ablation()


def make_supplemental_analyses():
    make_table_mvpa_models_mdls()
    make_table_prda_properties()
    make_table_rsa_properties()
    make_table_rsa_models()
    make_table_nlea_properties()
    make_table_nlea_properties_max()
    make_table_vwea_models()


if __name__ == "__main__":
    make_core_analyses()
    try:
        make_supplemental_analyses()
    except:
        print("not making all supplemental analyses")
