import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_results():
    return pd.read_csv("../outputs/results.csv")


def make_plot(table):
    mu = table.P.values[0]
    feature = table.FEATURE.values[0]
    sns.boxenplot(
        x="NETWORK",
        y="ACC",
        data=table,
        palette="crest",
        k_depth="proportion",
        outlier_prop=1e-20,
    )
    sns.swarmplot(x="NETWORK", y="ACC", data=table, color="black")
    plt.axhline(mu, color="black")
    plt.ylim([mu - 0.4, mu + 0.4])
    plt.title(feature.upper())
    plt.xticks()
    plt.savefig("../plots/bar/%s.jpg" % "_".join(feature.split()))
    plt.clf()


def main():
    results = load_results()
    for feature in results.FEATURE.unique():
        table = results[results.FEATURE == feature]
        make_plot(table)


if __name__ == "__main__":
    main()
