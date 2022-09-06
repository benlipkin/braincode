import itertools
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *


def load_probing_results():
    cache = "../../braincode/.cache"
    table = collections.defaultdict(list)
    for emb, prop in itertools.chain(
        itertools.product(BRAIN.items(), PROPS.items()),
        itertools.product(MODELS.items(), PROPS.items()),
    ):
        prefix = "prda" if "code" in emb[0] else "mvpa"
        score = np.tanh(
            np.load(f"{cache}/scores/{prefix}/score_{emb[0]}_{prop[0]}_FisherCorr.npy")
        )
        subjects = (
            np.tanh(
                np.load(
                    f"{cache}/scores/{prefix}/subjects_{emb[0]}_{prop[0]}_FisherCorr.npy"
                )
            )
            if prefix == "mvpa"
            else None
        )
        table["Analysis"].append(prefix)
        table["Embedding"].append(emb[1])
        table["Property"].append(prop[1])
        table["Score"].append(score)
        table["Subjects"].append(subjects)
    return pd.DataFrame(table)


def plot_brain_probing(data, bar_width=0.2):
    ax = plt.subplot(111)
    for i, emb in enumerate(data["Embedding"].unique()):
        samples = data[data["Embedding"] == emb]
        scores = samples["Score"]
        error = [arr.std() / np.sqrt(arr.size) for arr in samples["Subjects"]]
        r = (
            np.array([x + bar_width for x in r])
            if i
            else np.arange(len(scores)) - 0.5 * bar_width
        )
        color = [["navy"], ["royalblue"], ["maroon"], ["indianred"]][i]
        ax.bar(
            r,
            scores,
            yerr=error,
            color=color,
            width=bar_width,
            edgecolor="black",
            capsize=3,
            label=emb,
        )
    plt.xlabel("Code Property", fontweight="bold")
    plt.ylabel("Probing Score (Pearson R)", fontweight="bold")
    plt.legend()
    plt.ylim([0, 0.5])
    plt.xticks(
        [r + bar_width for r in range(len(scores))],
        data["Property"].unique(),
        rotation=45,
    )
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    plt.show()
    plt.close()


def plot_model_probing(data):
    for i, prop in enumerate(data["Property"].unique()):
        ax = plt.subplot(2, 2, i + 1)
        samples = data[data["Property"] == prop]
        baseline = samples[samples["Embedding"] == "Token Projection"].Score.values
        scores_nl = samples[samples["Embedding"].str.contains("NL")].Score.values
        scores_py = samples[samples["Embedding"].str.contains("Py")].Score.values
        n_params = [350e6, 2e9, 6e9, 16e9]
        plt.axhline(baseline, color="black", linestyle="--", label="Token Projection")
        plt.plot(n_params, scores_nl, "o-", color="forestgreen", label="NL")
        plt.plot(n_params, scores_py, "o-", color="darkgreen", label="Python")
        plt.title(prop, fontweight="bold")
        plt.xlabel("# of Model Parameters", fontweight="bold")
        plt.ylabel("Probing Score (Pearson R)", fontweight="bold")
        plt.legend()
        plt.xscale("log")
        plt.ylim([0.7, 1.0])
        for spine in ["left", "right", "top", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    data = load_probing_results()
    plot_brain_probing(data.loc[data["Analysis"] == "mvpa"])
    plot_model_probing(data.loc[data["Analysis"] == "prda"])


if __name__ == "__main__":
    main()
