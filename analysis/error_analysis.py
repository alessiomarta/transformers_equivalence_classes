import os
import argparse
from operator import itemgetter
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-path", type=str, required=True)
    parser.add_argument("--pretty-print", type=str, default="n", choices=["y", "n"])
    return parser.parse_args()


def main():
    args = parse_args()

    sns.set_theme(
        rc={
            "figure.figsize": (14, 7),
        }
    )
    for exp in os.listdir(args.res_path):
        if exp.startswith("sime"):
            all_stats = {"Iteration": [], "Mean difference": [], "Experiment": []}
            for res in os.listdir(os.path.join(args.res_path, exp)):
                if os.path.exists(
                    os.path.join(args.res_path, exp, res, "pred-stats.json")
                ):
                    exp_name = exp.split(os.path.sep)[-1]
                    if args.pretty_print == "y":
                        exp_name = "-".join(
                            itemgetter(*[0, 1])(exp.split(os.path.sep)[-1].split("-"))
                        )
                    stats = json.load(
                        open(
                            os.path.join(args.res_path, exp, res, "pred-stats.json"),
                            "r",
                        )
                    )
                    for k in stats:
                        all_stats["Iteration"].append(int(k))
                        all_stats["Mean difference"].append(1 - stats[k])
            pl = sns.lineplot(
                data=all_stats,
                x="Iteration",
                y="Mean difference",
                markers=True,
                dashes=False,
            )
            pl.set(
                ylim=(-0.1, 1.1),
                title=f"Mean difference in pre- and post-decoder predictions, for each iteration\nExperiment: {exp_name}",
            )
            figure = pl.get_figure()
            figure.savefig(os.path.join(args.res_path, exp, "error-stats.png"), dpi=400)
            plt.close()


if __name__ == "__main__":
    main()
