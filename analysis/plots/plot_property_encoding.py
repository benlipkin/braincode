import itertools
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *


def load_encoding_results():
    cache = "../../braincode/.cache"
    table = collections.defaultdict(list)
    for emb, prop in itertools.chain(
        itertools.product(BRAIN.items(), JOINT.items()),
        itertools.product(MODELS.items(), JOINT.items()),
        itertools.product(BRAIN.items(), CEILING.items()),
    ):
        prefix = "prea" if "code" in emb[0] else "nlea"
        prefix = "cnlea" if "test" in prop[0] else prefix
        score = np.tanh(
            np.load(f"{cache}/scores/{prefix}/score_{emb[0]}_{prop[0]}_FisherCorr.npy")
        )
        subjects = (
            np.tanh(
                np.load(
                    f"{cache}/scores/{prefix}/subjects_{emb[0]}_{prop[0]}_FisherCorr.npy"
                )
            )
            if prefix == "nlea"
            else np.array([0] * 24)
        )
        table["Analysis"].append(prefix)
        table["Embedding"].append(emb[1])
        table["Property"].append(prop[1])
        table["Score"].append(score)
        table["Subjects"].append(subjects)
    return pd.DataFrame(table)


def plot_encoding(data):
    ax = plt.subplot(111)
    samples = data[data.Property == "All Properties"]
    x = np.arange(samples.shape[0]).astype(float)
    x[4:] += 0.5
    colors = (
        ["navy", "royalblue", "maroon", "indianred", "silver"]
        + ["forestgreen"] * 4
        + ["darkgreen"] * 4
    )
    scores = samples["Score"]
    errors = [arr.std() / np.sqrt(arr.size) for arr in samples["Subjects"]]
    plt.bar(x, scores, yerr=errors, width=1, color=colors, edgecolor="black")
    ceilings = data[data.Property == "Ceiling"].Score.values
    for i, ceil in enumerate(ceilings):
        plt.plot([i - 0.5, i + 0.5], [ceil, ceil], "--", color="black", linewidth=2)
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    plt.xlabel("Code Representation", fontweight="bold")
    plt.ylabel("Encoding Score (Pearson R)", fontweight="bold")
    plt.xticks(x, samples["Embedding"].unique(), rotation=45)
    plt.ylim([0, 1])
    plt.show()


def main():
    data = load_encoding_results()
    plot_encoding(data)


if __name__ == "__main__":
    main()
