import itertools
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *


def load_encoding_results():
    cache = "../../braincode/.cache"
    table = collections.defaultdict(list)
    for region, model in itertools.product(BRAIN.items(), MODELS.items()):
        prefix = "nlea"
        score = np.tanh(
            np.load(
                f"{cache}/scores/{prefix}/score_{region[0]}_{model[0]}_FisherCorr.npy"
            )
        )
        subjects = np.tanh(
            np.load(
                f"{cache}/scores/{prefix}/subjects_{region[0]}_{model[0]}_FisherCorr.npy"
            )
        )
        table["Analysis"].append(prefix)
        table["Region"].append(region[1])
        table["Model"].append(model[1])
        table["Score"].append(score)
        table["Subjects"].append(subjects)
    return pd.DataFrame(table)


def plot_model_encoding_by_network(data):
    for i, region in enumerate(data["Region"].unique()):
        ax = plt.subplot(3, 2, i + 1)
        samples = data[data["Region"] == region]
        baseline = samples[samples["Model"] == "Token Projection"].Score.values
        samples_nl = samples[samples["Model"].str.contains("NL")]
        scores_nl = samples_nl.Score.values
        err_nl = [arr.std() / np.sqrt(arr.size) for arr in samples_nl["Subjects"]]
        samples_py = samples[samples["Model"].str.contains("Py")]
        scores_py = samples_py.Score.values
        err_py = [arr.std() / np.sqrt(arr.size) for arr in samples_py["Subjects"]]
        n_params = [350e6, 2e9, 6e9, 16e9]
        plt.axhline(baseline, color="black", linestyle="--", label="Token Projection")
        plt.errorbar(
            n_params, scores_nl, err_nl, color="forestgreen", capsize=2, label="NL"
        )
        plt.errorbar(
            n_params, scores_py, err_py, color="darkgreen", capsize=2, label="Python"
        )
        plt.title(region, fontweight="bold")
        plt.xlabel("# of Model Parameters", fontweight="bold") if i == 2 else None
        plt.ylabel("Encoding Score (Pearson R)", fontweight="bold") if i == 2 else None
        plt.xscale("log")
        plt.ylim([-0.1, 0.3])
        for spine in ["left", "right", "top", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
    ax = plt.subplot(3, 2, 5)
    plt.axhline(baseline, color="black", linestyle="--", label="Token Projection")
    plt.errorbar(
        n_params, scores_nl, err_nl, color="forestgreen", capsize=2, label="NL"
    )
    plt.errorbar(
        n_params, scores_py, err_py, color="darkgreen", capsize=2, label="Python"
    )
    h, l = ax.get_legend_handles_labels()
    ax.clear()
    ax.legend(h, l, loc="center left")
    ax.axis("off")
    plt.gcf().set_size_inches(6, 6)
    plt.tight_layout()
    plt.savefig(f"fig_enc_brain_models_byregion.png", bbox_inches="tight", dpi=600)
    plt.close()


def plot_model_encoding_by_corpus(data):
    for i, corpus in enumerate(["NL", "Python"]):
        ax = plt.subplot(2, 2, i + 1)
        samples = data[data["Model"].str.contains(corpus)]
        samples_md_lh = samples[samples["Region"] == "MD LH"]
        scores_md_lh = samples_md_lh.Score.values
        err_md_lh = [arr.std() / np.sqrt(arr.size) for arr in samples_md_lh.Subjects]
        samples_md_rh = samples[samples["Region"] == "MD RH"]
        scores_md_rh = samples_md_rh.Score.values
        err_md_rh = [arr.std() / np.sqrt(arr.size) for arr in samples_md_rh.Subjects]
        samples_lang_lh = samples[samples["Region"] == "Lang LH"]
        scores_lang_lh = samples_lang_lh.Score.values
        err_lang_lh = [
            arr.std() / np.sqrt(arr.size) for arr in samples_lang_lh.Subjects
        ]
        samples_lang_rh = samples[samples["Region"] == "Lang RH"]
        scores_lang_rh = samples_lang_rh.Score.values
        err_lang_rh = [
            arr.std() / np.sqrt(arr.size) for arr in samples_lang_rh.Subjects
        ]
        n_params = [350e6, 2e9, 6e9, 16e9]
        plt.errorbar(
            n_params,
            scores_md_lh,
            err_md_lh,
            color="navy",
            capsize=2,
            label="MD LH",
        )
        plt.errorbar(
            n_params,
            scores_md_rh,
            err_md_rh,
            color="royalblue",
            capsize=2,
            label="MD RH",
        )
        plt.errorbar(
            n_params,
            scores_lang_lh,
            err_lang_lh,
            color="maroon",
            capsize=2,
            label="Lang LH",
        )
        plt.errorbar(
            n_params,
            scores_lang_rh,
            err_lang_rh,
            color="indianred",
            capsize=2,
            label="Lang RH",
        )
        plt.title(corpus, fontweight="bold")
        plt.xlabel("# of Model Parameters", fontweight="bold")
        plt.ylabel("Encoding Score (Pearson R)", fontweight="bold")
        plt.xscale("log")
        plt.ylim([-0.1, 0.3])
        for spine in ["left", "right", "top", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
    ax = plt.subplot(2, 2, 3)
    plt.errorbar(
        n_params,
        scores_md_lh,
        err_md_lh,
        color="navy",
        capsize=2,
        label="MD LH",
    )
    plt.errorbar(
        n_params,
        scores_md_rh,
        err_md_rh,
        color="royalblue",
        capsize=2,
        label="MD RH",
    )
    plt.errorbar(
        n_params,
        scores_lang_lh,
        err_lang_lh,
        color="maroon",
        capsize=2,
        label="Lang LH",
    )
    plt.errorbar(
        n_params,
        scores_lang_rh,
        err_lang_rh,
        color="indianred",
        capsize=2,
        label="Lang RH",
    )
    h, l = ax.get_legend_handles_labels()
    ax.clear()
    ax.legend(h, l, loc="center left")
    ax.axis("off")
    plt.gcf().set_size_inches(6, 4)
    plt.tight_layout()
    plt.savefig(f"fig_enc_brain_models_bycorpus.png", bbox_inches="tight", dpi=600)
    plt.close()


def plot_model_encoding_correlations(data):
    for i, (region_a, region_b) in enumerate(
        itertools.combinations(data.Region.unique(), 2)
    ):
        samples_a = np.concatenate(data[data["Region"] == region_a]["Subjects"].values)
        samples_b = np.concatenate(data[data["Region"] == region_b]["Subjects"].values)
        ax = plt.subplot(2, 3, i + 1)
        plt.scatter(samples_a, samples_b, color="black", s=1)
        scale = 0.8
        plt.xlabel(region_a, fontweight="bold")
        plt.ylabel(region_b, fontweight="bold")
        plt.xlim([-scale, scale])
        plt.ylim([-scale, scale])
        for spine in ["left", "right", "top", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
    plt.tight_layout()
    plt.savefig(f"fig_enc_brain_models_regioncorrs.png", bbox_inches="tight", dpi=600)
    plt.close()


def main():
    data = load_encoding_results()
    plot_model_encoding_by_network(data)
    plot_model_encoding_by_corpus(data)
    plot_model_encoding_correlations(data)


if __name__ == "__main__":
    main()
