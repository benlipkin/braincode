import glob
import itertools
import pickle as pkl

import numpy as np
import pandas as pd


def get_subjects():
    return [
        int(f.split("/")[-1].split(".")[0])
        for f in sorted(glob.glob("../../../../braincode/inputs/neural_data/*.mat"))
        if "737" not in f
    ]


def get_data(subject, network, feature):
    with open(
        f"../../../../braincode/.cache/representations/mvpa/{network}_{feature}{subject}.pkl",
        "rb",
    ) as f:
        data = pkl.load(f)
    return data


def feature_map(feature):
    return {
        "content": "datatype",
        "lang": "variable_language",
        "tokens": "num_tokens",
        "lines": "num_runtime_steps",
    }[feature]


def main():
    subjects = get_subjects()
    networks = ["MD", "lang", "vis", "aud"]
    features = ["structure", "content", "lang", "tokens", "lines"]
    data_dict = {}
    for subject, network, feature in itertools.product(subjects, networks, features):
        data = get_data(subject, network, feature)
        if subject not in data_dict.keys():
            data_dict[subject] = {}
        if network not in data_dict[subject].keys():
            data_dict[subject][network] = {}
        data_dict[subject][network]["brain_reps"] = data["X"]
        data_dict[subject][network]["cv_group_folds"] = data["runs"].flatten()
        if feature == "structure":
            data_dict[subject][network]["conditional"] = (
                data["y"].flatten() == 1
            ).astype(int)
            data_dict[subject][network]["iteration"] = (
                data["y"].flatten() == 0
            ).astype(int)
        else:
            data_dict[subject][network][feature_map(feature)] = data["y"].flatten()
    with open("aggregated_data.pkl", "wb") as f:
        pkl.dump(data_dict, f)


if __name__ == "__main__":
    main()
