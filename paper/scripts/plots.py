import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(dataset):
    return pd.read_csv(f"../tables/raw/{dataset}.csv")


def update_names(data):
    data.loc[data.Feature == "MD+lang+vis+aud", "Feature"] = "MD+LVA"
    data.loc[data.Feature == "MD+lang+vis", "Feature"] = "MD+LV"
    data.loc[data.Feature == "MD+lang", "Feature"] = "MD+L"
    data.loc[data.Feature == "MD", "Feature"] = "MD"
    data.loc[data.Feature == "lang", "Feature"] = "Language"
    data.loc[data.Feature == "vis", "Feature"] = "Visual"
    data.loc[data.Feature == "aud", "Feature"] = "Auditory"
    data.loc[data.Feature == "random", "Feature"] = "Random Embedding"  #
    data.loc[data.Feature == "bow", "Feature"] = "Bag Of Words"
    data.loc[data.Feature == "tfidf", "Feature"] = "TF-IDF"
    data.loc[data.Feature == "seq2seq", "Feature"] = "Seq2Seq"
    data.loc[data.Feature == "xlnet", "Feature"] = "XLNet"
    data.loc[data.Feature == "ct", "Feature"] = "CodeTransformer"
    data.loc[data.Feature == "codeberta", "Feature"] = "CodeBERTa"
    data.loc[data.Target == "code", "Target"] = "Code vs. Sentence"
    data.loc[data.Target == "lang", "Target"] = "Variable Language"
    data.loc[data.Target == "content", "Target"] = "Data Type"
    data.loc[data.Target == "structure", "Target"] = "Control Flow"
    data.loc[data.Target == "lines", "Target"] = "Runtime Steps"
    data.loc[data.Target == "nodes", "Target"] = "Node Count"
    data.loc[data.Target == "tokens", "Target"] = "Token Count"
    data.loc[data.Target == "halstead", "Target"] = "Halstead Difficulty"  #
    data.loc[data.Target == "cyclomatic", "Target"] = "Cyclomatic Complexity"  #
    data.loc[data.Target == "random", "Target"] = "Random Embedding"  #
    data.loc[data.Target == "bow", "Target"] = "Bag Of Words"
    data.loc[data.Target == "tfidf", "Target"] = "TF-IDF"
    data.loc[data.Target == "seq2seq", "Target"] = "Seq2Seq"
    data.loc[data.Target == "xlnet", "Target"] = "XLNet"
    data.loc[data.Target == "ct", "Target"] = "CodeTransformer"
    data.loc[data.Target == "codeberta", "Target"] = "CodeBERTa"
    return data


def make_base_plot(data, dataset):
    bar_width = 0.16
    cidx = 0
    ax = plt.subplot(111)
    for i, rep in enumerate(data["Feature"].unique()):
        samples = data[data["Feature"] == rep]
        scores = samples["Score"].values
        error = samples["95CI"].values
        if not i:
            r = np.arange(len(scores)) - 0.5 * bar_width
        else:
            r = np.array([x + bar_width for x in r])
        if "ablation" in dataset:
            color = np.array(
                [1.0 - (cidx * 0.3), 0.05 + (cidx * 0.15), 0 + (0.3 * cidx)]
            )
        else:
            color = np.array(
                [0.1 + (cidx * 0.30), 0.5 + (cidx * 0.15), 0.9 - (cidx * 0.30)]
            )
        cidx += 1
        ax.bar(
            r,
            scores,
            yerr=error,
            color=color,
            width=bar_width,
            edgecolor="black",
            label=rep,
            capsize=2,
        )
    plt.xticks(
        [r + bar_width for r in range(len(scores))],
        data["Target"].unique(),
        rotation=45,
    )
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    return ax


def individual_formatting(ax, dataset):
    data = load_data(dataset)
    cfg = {
        "mvpa_properties_cls": {
            "xlabel": "Code Properties",
            "ylabel": "Classification Accuracy (%)",
            "ylim": [0, 1],
            "yticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "size": (6, 4),
            "sig_y": 0.025,
            "legend_loc": (0.75, 0.90),
        },
        "mvpa_properties_rgr": {
            "xlabel": "Code Properties",
            "ylabel": "Pearson Correlation (r)",
            "ylim": [-0.1, 0.45],
            "yticks": [0, 0.1, 0.2, 0.3, 0.4],
            "size": (3, 4),
            "sig_y": 0.0125,
            "legend_loc": (0.77, 1.00),
        },
        "mvpa_models": {
            "xlabel": "Code Model",
            "ylabel": "Rank Accuracy (%)",
            "ylim": [0.45, 0.65],
            "yticks": [0.45, 0.50, 0.55, 0.60, 0.65],
            "size": (8, 4),
            "sig_y": 0.45 + 0.0025,
            "legend_loc": (0.80, 1.03),
        },
    }
    dataset = dataset.replace("_ablation", "")
    dataset_cfg = cfg[dataset]
    plt.xlabel(dataset_cfg["xlabel"], fontweight="bold")
    plt.ylabel(dataset_cfg["ylabel"], fontweight="bold")
    plt.ylim(dataset_cfg["ylim"])
    plt.yticks(dataset_cfg["yticks"], labels=[f"{t}" for t in dataset_cfg["yticks"]])
    plt.legend(loc="center left", bbox_to_anchor=dataset_cfg["legend_loc"])
    for i, target in enumerate(data.Target.unique()):
        baseline = data[data.Target == target]["Null Mean"].mean()
        ax.plot([i - 0.25, i + 0.6], [baseline, baseline], "--", color="0.25")
    plt.gcf().set_size_inches(*dataset_cfg["size"])
    x_start = -0.12
    for target in data.Target.unique():
        samples = data[data.Target == target]
        sigs = samples["h (corrected)"] == 1
        for i, sig in enumerate(sigs):
            x = x_start + 0.16 * i
            if sig:
                plt.annotate("*", (x, dataset_cfg["sig_y"]))
        x_start += 1
    return ax


def plot_data(dataset):
    data = load_data(dataset)
    data = update_names(data)
    ax = make_base_plot(data, dataset)
    ax = individual_formatting(ax, dataset)
    plt.savefig(f"../plots/{dataset}.jpg", bbox_inches="tight")
    plt.close()


def main():
    datasets = [
        "mvpa_properties_cls",
        "mvpa_properties_rgr",
        "mvpa_models",
        "mvpa_properties_cls_ablation",
        "mvpa_properties_rgr_ablation",
        "mvpa_models_ablation",
    ]
    for dataset in datasets:
        plot_data(dataset)


if __name__ == "__main__":
    main()