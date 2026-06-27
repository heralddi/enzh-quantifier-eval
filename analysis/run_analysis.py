"""Bayesian hierarchical logistic regression on the Qwen scoring results.

Model: congruent ~ phenomenon * language with a random intercept per item
pair, estimated with PyMC. Treatment coding, reference cell scalar_implicature
in English. Outputs an ArviZ posterior summary CSV and a forest plot of the
fixed effects.

Usage:
    python analysis/run_analysis.py --scores results/scores.csv --outdir results
"""

import argparse
import os

import numpy as np
import pandas as pd

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
REFERENCE_PHENOMENON = "scalar_implicature"
SEED = 20260703


def build_design(df):
    phens = sorted(df["phenomenon"].unique())
    others = [p for p in phens if p != REFERENCE_PHENOMENON]
    cols = {"intercept": np.ones(len(df))}
    for p in others:
        cols[f"phen[{p}]"] = (df["phenomenon"] == p).to_numpy(dtype=float)
    cols["lang[zh]"] = (df["language"] == "zh").to_numpy(dtype=float)
    for p in others:
        cols[f"phen[{p}]:lang[zh]"] = (
            (df["phenomenon"] == p) & (df["language"] == "zh")
        ).to_numpy(dtype=float)
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names


def fit(df, draws=2000, tune=1000, chains=2):
    import pymc as pm

    X, coef_names = build_design(df)
    y = df["congruent"].to_numpy(dtype=int)
    pair_levels, pair_idx = np.unique(df["pair_id"], return_inverse=True)

    coords = {"coef": coef_names, "pair": pair_levels}
    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", 0.0, 1.5, dims="coef")
        sigma_pair = pm.HalfNormal("sigma_pair", 1.0)
        z_pair = pm.Normal("z_pair", 0.0, 1.0, dims="pair")
        eta = pm.math.dot(X, beta) + sigma_pair * z_pair[pair_idx]
        pm.Bernoulli("y", logit_p=eta, observed=y)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains,
            target_accept=0.9,
            random_seed=SEED,
            progressbar=False,
        )
    return idata, coef_names


def posterior_summary(idata):
    import arviz as az

    summ = az.summary(idata, var_names=["beta", "sigma_pair"])
    return summ


def forest_plot(idata, coef_names, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    beta = np.asarray(idata.posterior["beta"])  # chains x draws x coef
    flat = beta.reshape(-1, beta.shape[-1])
    means = flat.mean(axis=0)
    lo, hi = np.percentile(flat, [5.5, 94.5], axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ypos = np.arange(len(coef_names))[::-1]
    ax.hlines(ypos, lo, hi, color="steelblue", lw=2)
    ax.plot(means, ypos, "o", color="steelblue")
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(coef_names)
    ax.set_xlabel("log-odds of a congruent choice (posterior mean, 89% interval)")
    ax.set_title("Fixed effects, congruent ~ phenomenon * language\n"
                 "reference cell: scalar_implicature, English (Qwen2.5-0.5B)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    df = df[df["model"] == MODEL_NAME].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no rows for model {MODEL_NAME} in {args.scores}")
    print(f"fitting on {len(df)} observations, {df['pair_id'].nunique()} pairs")

    idata, coef_names = fit(df, args.draws, args.tune, args.chains)
    summ = posterior_summary(idata)
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "posterior_summary.csv")
    summ.to_csv(csv_path)
    print(summ.to_string())
    print(f"wrote {csv_path}")

    png_path = os.path.join(args.outdir, "forest_fixed_effects.png")
    forest_plot(idata, coef_names, png_path)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
