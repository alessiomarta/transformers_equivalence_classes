import pandas as pd
import numpy as np
import seaborn as sns
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, required = True,
                    help = "Result file.")
args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists("../figures"):
        os.makedirs("../figures")

    results = pd.read_csv(args.file, index_col = 0)
    filename = args.file.split(".")[0]

    if "explore-token" in results.columns:
        alternative_token_ranks = results.groupby(["iteration", "file-name", "algorithm", "explore-token"]).agg(list)
        alternative_token_ranks['alternative-token-rank'] = alternative_token_ranks['alternative-token-proba'].apply(lambda L: 5 - np.argsort(L))
        alternative_token_ranks = alternative_token_ranks.explode(["alternative-token-rank", "alternative-token"]).set_index("alternative-token", append = True)['alternative-token-rank']
        results = pd.merge(results, alternative_token_ranks, how = "inner", left_on=["iteration", "file-name", "algorithm", "explore-token", "alternative-token"], right_index = True)

    g = sns.FacetGrid(data = results, col = "algorithm", sharey=True)
    y = "modified-pred-proba" if "explore-token" in results.columns else "modified-original-proba"
    g.map(sns.lineplot, "iteration", y, errorbar = ("ci", 95), weights = results["alternative-token-proba"])
    g.add_legend()
    g.savefig(f"../figures/{filename}-general.pdf")

    if "explore-token" in results.columns:
        f= sns.FacetGrid(data = results, col = "algorithm", hue = "alternative-token-rank", sharey=True)
        f.map(sns.lineplot, "iteration", "modified-pred-proba", errorbar = ("ci", 0))
        f.add_legend()
    else:
        melted_df = pd.melt(results, id_vars=["iteration", "algorithm"], value_vars=["modified-pred-proba", "modified-original-proba"])
        f = sns.FacetGrid(data = melted_df, col="algorithm", hue = "variable", sharey = True)
        f.map(sns.lineplot, "iteration", "value", errorbar = ("ci", 0))
        f.add_legend()

    f.savefig(f"../figures/{filename}-specific.pdf")

    h= sns.FacetGrid(data = results, col = "algorithm", sharey=True)
    h.map(sns.lineplot, "iteration", y, estimator = "var", errorbar = ("ci", 0))
    h.add_legend()
    h.savefig(f"../figures/{filename}-variance.pdf")

    results['same-label'] = results.apply(lambda row: int(row['original-pred'] == row['alternative-pred']), axis = 1)
    l = sns.FacetGrid(data = results, col = "algorithm", sharey = True)
    l.map(sns.lineplot, "iteration", "same-label", errorbar = ("ci", 0))
    l.add_legend()
    l.savefig(f"../figures/{filename}-frequency.pdf")