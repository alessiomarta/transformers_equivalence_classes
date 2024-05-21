import argparse
import os
import torch
import pickle
import numpy as np


def main():

    # python time_analysis.py --dir ../../res/1-sqrtmax-1-sqrtmax/input-space-exploration --iter 1000

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--iter", required=True, type=int)
    args = parser.parse_args()

    times = []

    for experiment in os.listdir(args.dir):
        if os.path.isdir(os.path.join(args.dir, experiment)) and "bert" in experiment:
            for folder in os.listdir(os.path.join(args.dir, experiment)):
                filepath = os.path.join(
                    args.dir, experiment, folder, f"{args.iter}.pkl"
                )
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "rb") as outp:
                            d = pickle.load(outp)
                        print(filepath)
                        times.append(d["time"])
                    except:
                        print("Corrupted file")
                        continue

    print(np.mean(times), np.std(times))


if __name__ == "__main__":
    main()
