import json
import os
import pickle as pkl
import sys

import numpy as np


def clean_cache(pth):
    files = os.listdir(pth)
    print('Clear path? {}'.format(pth))
    inp = input()
    if 'y' in inp.lower() or '1' in inp.lower():
        for f in files:
            if "score" in f and ".npy" in f:
                print('Clearing {}'.format(os.path.join(pth, f)))
                os.remove(os.path.join(pth, f))


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
    choice = int(sys.argv[2])
    
    if choice == 1:
        clean_cache(pth)
    elif choice == 2:
        print_scores(pth)
