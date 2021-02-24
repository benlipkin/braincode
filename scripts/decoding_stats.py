import pandas as pd


def get_data():
    return pd.read_csv("../outputs/results.csv")


def summarize(data):
    for network in data.NETWORK.unique():
        features = data[data.NETWORK == network]
        for feature in features.FEATURE.unique():
            subjects = features[features.FEATURE == feature]
            print("%s\t%s\t%.3f" % (network, feature, subjects.ACC.mean()))
            # expand to run t-test and save ../outputs/stats.csv


def main():
    summarize(get_data())


if __name__ == "__main__":
    main()
