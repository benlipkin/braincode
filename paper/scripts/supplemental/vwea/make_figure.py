import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data():
    return pd.read_csv(f"../../../tables/raw/vwea_models.csv")


def update_names(data):
    data.loc[data.Feature == "MD", "Feature"] = "MD"
    data.loc[data.Feature == "lang", "Feature"] = "Language"
    data.loc[data.Target == "projection", "Target"] = "Token Projection"
    data.loc[data.Target == "bow", "Target"] = "Bag Of Words"
    data.loc[data.Target == "tfidf", "Target"] = "TF-IDF"
    data.loc[data.Target == "seq2seq", "Target"] = "Seq2Seq"
    data.loc[data.Target == "xlnet", "Target"] = "XLNet"
    data.loc[data.Target == "transformer", "Target"] = "CodeTransformer"
    data.loc[data.Target == "roberta", "Target"] = "CodeBERTa"
    data.loc[data.Target == "bert", "Target"] = "CodeBERT"
    data.loc[data.Target == "gpt2", "Target"] = "CodeGPT"
    return data


def make_figure(data):
    bar_width = 0.08
    ax = plt.subplot(111)
    for i, rep in enumerate(data["Target"].unique()):
        samples = data[data["Target"] == rep]
        scores = samples["Score"].values
        error = samples["95CI"].values
        if not i:
            r = np.arange(len(scores)) - 3 * bar_width
        else:
            r = np.array([x + bar_width for x in r])
        color = color = [
            "tab:brown",
            "tab:red",
            "tab:orange",
            "tab:olive",
            "tab:green",
            "tab:cyan",
            "tab:blue",
            "tab:purple",
            "tab:pink",
        ][i]
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
        data["Feature"].unique(),
        rotation=45,
    )
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.plot([-0.35, 1.65], [0, 0], "--", color="0.25")
    plt.xlabel("Brain Network", fontweight="bold")
    plt.ylabel("Rank Accuracy (%)", fontweight="bold")
    plt.ylim([0.48, 0.60])
    plt.legend(loc="center left", bbox_to_anchor=[0.80, 0.95])
    plt.gcf().set_size_inches([8, 4])
    x_start = -3 * bar_width - 0.015
    for target in data.Feature.unique():
        samples = data[data.Feature == target]
        sigs = samples["h (corrected)"] == 1
        for i, sig in enumerate(sigs):
            x = x_start + bar_width * i
            if sig:
                plt.annotate("*", (x, 0.49))
        x_start += 1
    return ax


def main():
    data = load_data()
    data = update_names(data)
    ax = make_figure(data)
    plt.savefig(f"vwea_models.png", bbox_inches="tight", dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
