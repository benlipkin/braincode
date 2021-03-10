import pandas as pd
from scipy.stats import ttest_1samp

if __name__ == "__main__":
    ostream = "NETWORK,FEATURE,N,NULL,MEAN,T,PVAL,H\n"
    data = pd.read_csv("../outputs/results.csv")
    for network in data.NETWORK.unique():
        features = data[data.NETWORK == network]
        for feature in features.FEATURE.unique():
            subjects = features[features.FEATURE == feature]
            null = subjects.P.values[0]
            samples = subjects.ACC
            tstat, pval = ttest_1samp(samples, null)
            ostream += "%s,%s,%d,%f,%f,%f,%f,%d\n" % (
                network,
                feature,
                samples.size,
                null,
                samples.mean(),
                tstat,
                pval,
                int(pval < 0.05),
            )
    with open("../outputs/stats.csv", "w") as f:
        f.write(ostream)
