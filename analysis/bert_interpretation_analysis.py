import os
import argparse
from collections import defaultdict
from operator import itemgetter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=str, required=True)
    parser.add_argument("--pretty-print", type=str, default="n", choices=["y", "n"])
    return parser.parse_args()


def main():
    args = parse_args()
    pre_eq_class_wrds = []
    cap_eq_class_wrds = []
    sns.set_theme(
        rc={
            "figure.figsize": (12, 6),
        }
    )
    exp_name = args.json_dir.split(os.path.sep)[-1]
    if args.pretty_print == "y":
        exp_name = "-".join(
            itemgetter(*[0, 1])(args.json_dir.split(os.path.sep)[-1].split("-"))
        )
    for res in os.listdir(args.json_dir):
        if os.path.isdir(os.path.join(args.json_dir, res)):
            files = [
                filename
                for filename in os.listdir(
                    os.path.join(args.json_dir, res, "interpretation")
                )
                if os.path.isfile(
                    os.path.join(args.json_dir, res, "interpretation", filename)
                )
                and filename.lower().endswith("-stats.json")
            ]
            pred_p = {"Predicted token": [], "Iteration": [], "Prediction value": []}
            eq_c_p = defaultdict(dict)
            for j_file in tqdm(files, desc=res):
                stats = json.load(
                    open(
                        os.path.join(args.json_dir, res, "interpretation", j_file), "r"
                    )
                )
                for el in stats["cap-probas"]:
                    pred_p["Predicted token"].append(el[0])
                    pred_p["Iteration"].append(int(j_file.split("-")[0]))
                    pred_p["Prediction value"].append(el[1])

                pre_eq_class_wrds_keys = [
                    k for k in stats.keys() if "pre-cap-probas-" in k
                ]
                for k in pre_eq_class_wrds_keys:
                    pre_eq_class_wrds.append(
                        sorted([el[1] for el in stats[k]], reverse=True)
                    )
                cap_eq_class_wrds_keys = [
                    k
                    for k in stats.keys()
                    if "cap-probas-" in k
                    and "pre-cap-probas-" not in k
                    and k != "cap-probas-mod"
                ]
                for k in cap_eq_class_wrds_keys:
                    for el in stats[k]:
                        try:
                            eq_c_p[k.split("-")[-1]]["Predicted token"].append(el[0])
                            eq_c_p[k.split("-")[-1]]["Iteration"].append(
                                int(j_file.split("-")[0])
                            )
                            eq_c_p[k.split("-")[-1]]["Prediction value"].append(el[1])
                        except KeyError:
                            eq_c_p[k.split("-")[-1]] = {
                                "Predicted token": [],
                                "Iteration": [],
                                "Prediction value": [],
                            }
                            eq_c_p[k.split("-")[-1]]["Predicted token"].append(el[0])
                            eq_c_p[k.split("-")[-1]]["Iteration"].append(
                                int(j_file.split("-")[0])
                            )
                            eq_c_p[k.split("-")[-1]]["Prediction value"].append(el[1])
                    cap_eq_class_wrds.append(
                        sorted([el[1] for el in stats[k]], reverse=True)
                    )

            df = pd.DataFrame.from_dict(pred_p)
            top_sums = list(
                df.groupby("Predicted token")
                .sum("Prediction value")
                .sort_values("Prediction value", ascending=False)[:5]
                .index
            )
            top_means = list(
                df.groupby("Predicted token")
                .mean("Prediction value")
                .sort_values("Prediction value", ascending=False)[:5]
                .index
            )
            top_preds = set(
                df.loc[df.groupby("Iteration")["Prediction value"].idxmax()][
                    "Predicted token"
                ].values
            )
            df = df[
                df["Predicted token"].isin(top_preds.union(top_means).union(top_sums))
            ].sort_values(by="Iteration")

            pl = sns.lineplot(
                data=df,
                x="Iteration",
                y="Prediction value",
                hue="Predicted token",
                style="Predicted token",
                markers=True,
                dashes=False,
            )
            pl.set(
                title=f"Predictions values, for each iteration\nExperiment {exp_name} on {res}"
            )
            box = pl.get_position()
            pl.set_position(
                [box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85]
            )
            sns.move_legend(pl, "lower center", bbox_to_anchor=(0.5, -0.3), ncol=7)
            figure = pl.get_figure()
            figure.savefig(os.path.join(args.json_dir, res, "probas.png"), dpi=400)
            plt.close()

            # probabilities plots
            for k in eq_c_p:
                df = pd.DataFrame.from_dict(eq_c_p[k])
                top_sums = list(
                    df.groupby("Predicted token")
                    .sum("Prediction value")
                    .sort_values("Prediction value", ascending=False)[:5]
                    .index
                )
                top_means = list(
                    df.groupby("Predicted token")
                    .mean("Prediction value")
                    .sort_values("Prediction value", ascending=False)[:5]
                    .index
                )
                top_preds = set(
                    df.loc[df.groupby("Iteration")["Prediction value"].idxmax()][
                        "Predicted token"
                    ].values
                )
                df = df[
                    df["Predicted token"].isin(
                        top_preds.union(top_means).union(top_sums)
                    )
                ].sort_values(by="Iteration")

                pl = sns.lineplot(
                    data=df,
                    x="Iteration",
                    y="Prediction value",
                    hue="Predicted token",
                    style="Predicted token",
                    markers=True,
                    dashes=False,
                )
                pl.set(
                    title=f"Top predicted tokens and their respective prediction values, for each iteration\nExperiment {exp_name} on {res}"
                )
                box = pl.get_position()
                pl.set_position(
                    [box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85]
                )
                sns.move_legend(pl, "lower center", bbox_to_anchor=(0.5, -0.3), ncol=7)
                figure = pl.get_figure()
                figure.savefig(
                    os.path.join(args.json_dir, res, f"{k}-probas.png"), dpi=400
                )
                plt.close()


if __name__ == "__main__":
    main()
