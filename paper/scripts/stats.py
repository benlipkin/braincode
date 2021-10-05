import itertools

import pandas as pd
import scipy.stats as st
from mne.stats import fdr_correction
from plots import update_names


def load_data(dataset):
    return pd.read_csv(f"../tables/raw/{dataset}.csv")


def calc_stats(dataset, reverse_grouping=False):
    groupings, S1, S2, tval, pval = [], [], [], [], []
    data = load_data(dataset)
    data = update_names(data)
    if reverse_grouping:
        data = data.rename(columns={"Feature": "Temp"})
        data = data.rename(columns={"Target": "Feature"})
        data = data.rename(columns={"Temp": "Target"})
    for grouping in data["Target"].unique():
        samples = data[data["Target"] == grouping]
        pairings = samples["Feature"].unique()
        for p1, p2 in itertools.combinations(pairings, 2):
            s1 = samples[samples["Feature"] == p1].iloc[0, 2:].values
            s2 = samples[samples["Feature"] == p2].iloc[0, 2:].values
            t, p = st.ttest_ind(s1, s2, equal_var=False)
            groupings.append(grouping)
            S1.append(p1)
            S2.append(p2)
            tval.append(t)
            pval.append(p)
    h_corrected, pval_corrected = fdr_correction(pval)
    stats = pd.DataFrame(
        {
            "Grouping": groupings,
            "S1": S1,
            "S2": S2,
            "t": tval,
            "p": pval,
            "p (corrected)": pval_corrected,
            "h (corrected)": h_corrected.astype("int"),
        }
    )
    if reverse_grouping:
        dataset = f"{dataset}_crossed"
    stats.to_csv(f"../stats/raw/{dataset}_stats.csv", index=False)


def calc_anova(dataset, reverse_grouping=False):
    groupings, fval, pval = [], [], []
    data = load_data(dataset)
    data = update_names(data)
    if reverse_grouping:
        data = data.rename(columns={"Feature": "Temp"})
        data = data.rename(columns={"Target": "Feature"})
        data = data.rename(columns={"Temp": "Target"})
    for grouping in data["Target"].unique():
        samples = data[data["Target"] == grouping]
        samples = samples.iloc[:, 2:].values
        f, p = st.f_oneway(*tuple(samples))
        groupings.append(grouping)
        fval.append(f)
        pval.append(p)
    h_corrected, pval_corrected = fdr_correction(pval)
    stats = pd.DataFrame(
        {
            "Grouping": groupings,
            "f": fval,
            "p": pval,
            "p (corrected)": pval_corrected,
            "h (corrected)": h_corrected.astype("int"),
        }
    )
    if reverse_grouping:
        dataset = f"{dataset}_crossed"
    dataset = f"{dataset}_anova"
    stats.to_csv(f"../stats/raw/{dataset}_stats.csv", index=False)


def main():
    datasets = [
        "mvpa_properties_all",
        "mvpa_models",
        "mvpa_properties_all_ablation",
        "mvpa_models_ablation",
    ]
    for dataset in datasets:
        calc_stats(f"{dataset}_subjects")
        calc_anova(f"{dataset}_subjects")
        if "models" in dataset:
            calc_stats(f"{dataset}_subjects", reverse_grouping=True)
            calc_anova(f"{dataset}_subjects", reverse_grouping=True)
        calc_stats("mvpa_properties_rgr", reverse_grouping=True)


if __name__ == "__main__":
    main()
