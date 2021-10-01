import pandas as pd
from plots import update_names


def load_data(dataset):
    return pd.read_csv(f"../tables/raw/{dataset}.csv")


def add_baselines(data):
    data["Empirical Baseline"] = 0
    for benchmark in data["Target"].unique():
        baseline = data[data["Target"] == benchmark]["Null Mean"].mean()
        data.loc[(data["Target"] == benchmark), "Empirical Baseline"] = baseline
    map = lambda x: f"{x:0.2f}"
    data["Empirical Baseline"] = data["Empirical Baseline"].apply(map)
    data.loc[(data["Empirical Baseline"] == "-0.00"), "Empirical Baseline"] = "0.00"
    return data


def make_pivot_table(data, dataset):
    if "prda" in dataset:
        index = "Feature"
        columns = ["Target", "Empirical Baseline"]
    else:
        index = ["Target", "Empirical Baseline"]
        columns = "Feature"
    return pd.pivot_table(
        data=data,
        index=index,
        columns=columns,
        values="Score",
    )


def reorder_columns(data, dataset):
    if "prda" in dataset:
        data = data.iloc[[3, 0, 5, 4, 6, 2, 1], :]
    elif "ablation" in dataset:
        pass
    else:
        data = data.iloc[:, [2, 1, 3, 0]]
    if "models" in dataset:
        data = data.iloc[[3, 0, 5, 4, 6, 2, 1], :]
    return data


def format_scores(data, dataset):
    map = lambda x: f"{x:0.2f}"
    for i, row in data.iterrows():
        if "prda" not in dataset and "rgr" not in dataset and "static" not in dataset:
            baseline = float(row.name[1])
            diff = row - baseline
            row = row.apply(map).values + " (+" + diff.apply(map).values + ")"
        else:
            row = row.apply(map).values
        row = [s.replace("+-", "-") for s in row]
        data.loc[i] = row
    if "rgr" in dataset or "static" in dataset:
        data = data.reset_index(level=1, drop=True)
    return data


def format_latex(latex, dataset):
    latex = latex.replace("{lllll}", "{l||llll}")
    latex = latex.replace("{llllll}", "{l||l|llll}")
    latex = latex.replace("{llllllll}", "{l||lllllll}")
    latex = latex.replace("{lllllllll}", "{l||l|lllllll}")
    latex = latex.replace("& Feature", "Feature &")
    if "prda" in dataset:
        latex = latex.replace("Feature", "Model Representation")
    else:
        latex = latex.replace("Feature", "Brain Representation")
    if "mvpa" in dataset and "model" in dataset:
        latex = latex.replace("Target", "Code Models")
    else:
        latex = latex.replace("Target", "Code Properties")
    return latex


def make_table(dataset):
    data = load_data(dataset)
    data = update_names(data)
    data = add_baselines(data)
    data = make_pivot_table(data, dataset)
    data = reorder_columns(data, dataset)
    data = format_scores(data, dataset)
    latex = format_latex(data.to_latex(), dataset)
    with open(f"../tables/latex/{dataset}.tex", "w") as f:
        f.write(latex)


def make_latex_table(dataset, type):
    dataset = f"{dataset}_{type}stats"
    data = pd.read_csv(f"../stats/raw/{dataset}.csv")
    data = data[data["h (corrected)"] == 1].iloc[:, :-1]
    if data.size:
        s = "Brain Region"
        if "crossed" in dataset:
            grouping = "Brain Region"
            s = "Code Model"
        elif "properties" in dataset:
            grouping = "Code Property"
        elif "model" in dataset:
            grouping = "Code Model"
        data = data.rename(columns={"Grouping": grouping})
        if "anova" not in dataset:
            data = data.rename(columns={"S1": f"{s} A", "S2": f"{s} B"})
        else:
            data = data.rename(columns={"f": "F"})
        data = data.set_index(grouping)
        latex = data.to_latex()
        latex = latex.replace("{lrrr}", "{l||rrr}")
        latex = latex.replace("{lllrrr}", "{l||ll|rrr}")
        with open(f"../stats/latex/{dataset}.tex", "w") as f:
            f.write(latex)


def main():
    datasets = [
        "mvpa_properties_cls",
        "mvpa_properties_rgr",
        "mvpa_models",
        "mvpa_properties_cls_ablation",
        "mvpa_properties_rgr_ablation",
        "mvpa_models_ablation",
        "prda_properties",
        "mvpa_properties_static",
    ]
    for dataset in datasets:
        make_table(dataset)
    datasets_stats = [
        "mvpa_properties_all_subjects",
        "mvpa_models_subjects",
        "mvpa_models_subjects_crossed",
        "mvpa_properties_all_ablation_subjects",
        "mvpa_models_ablation_subjects",
        "mvpa_models_ablation_subjects_crossed",
    ]
    for dataset in datasets_stats:
        make_latex_table(dataset, "")
        make_latex_table(dataset, "anova_")


if __name__ == "__main__":
    main()