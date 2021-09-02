import sys
import os
import pickle as pkl
import json
import numpy as np

def print_scores(pth):
    scores = {}
    files = os.listdir(pth)
    for f in files:
        if "score" in f and ".npy" in f:
            names = "".join(f.split(".npy")[:-1]).split("_")
            if len(names) == 3:
                feature = names[1]
                target = names[2]
                scores_np = np.load(os.path.join(pth, f))
                if feature not in scores:
                    scores[feature] = {}
                scores[feature][target] = scores_np.item()
    
    print("Model-wise scores: \n{}".format(json.dumps(scores, indent=2)))


if __name__ == "__main__":
    pth = sys.argv[1]
    print_scores(pth)