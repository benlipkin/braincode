import json
import os
import pickle as pkl
import sys

import numpy as np


def clean_cache(base_pth, choice):
    def _clean_cache(choice):
        if choice == 2:
            folder_name = "scores"
        elif choice == 3:
            folder_name = "representations"
        elif choice == 4:
            folder_name = "profiler"
        
        pth = os.path.join(base_pth, folder_name)
        print("Clear path? {}".format(pth))
        inp = input()
        if "y" in inp.lower() or "1" in inp.lower():
            for dir, _, filenames in os.walk(pth):
                for f in filenames:
                    if folder_name == "scores":
                        condition = "score" in f or ".npy" in f
                    elif folder_name == "representations":
                        condition = "pkl" in f
                    elif folder_name == "profiler":
                        condition = ".lprof" in f or ".py" in f
                    
                    if condition:
                        print("Clearing {}".format(os.path.join(pth, dir, f)))
                        os.remove(os.path.join(pth, dir, f))
    # clear scores
    if choice == 0:
        for i in [2, 3, 4]:
            _clean_cache(i)
    else:
        _clean_cache(choice)
    

def print_scores(base_pth):
    pth = os.path.join(base_pth, "scores")
    scores = {}
    folders = os.listdir(pth)
    for ff in folders:
        files = os.listdir(os.path.join(pth, ff))
        for f in files:
            if "score" in f and ".npy" in f:
                names = "".join(f.split(".npy")[:-1]).split("_")
                if len(names) == 3:
                    feature = names[1]
                    target = names[2]
                    scores_np = np.load(os.path.join(pth, ff, f))
                    if feature not in scores:
                        scores[feature] = {}
                    scores[feature][target] = scores_np.item()

    print("Model-wise scores: \n{}".format(json.dumps(scores, indent=2)))


if __name__ == "__main__":
    pth = sys.argv[1]
    choice = int(sys.argv[2])

    if choice == 1:
        print_scores(pth)
    elif choice in [0, 2, 3, 4]:
        clean_cache(pth, choice)
