"""Aggregate results/scores.csv into accuracy tables.

Usage:
    python -m src.summarise --scores results/scores.csv --out results/summary.md
"""

import argparse

import pandas as pd


def accuracy_table(df):
    tab = (
        df.groupby(["model", "phenomenon", "language"])["congruent"]
        .agg(["mean", "count"])
        .reset_index()
    )
    tab["accuracy"] = (tab["mean"] * 100).round(1)
    return tab.drop(columns=["mean"])


def to_markdown(df):
    lines = ["# Scoring summary", ""]
    lines.append("Chance level is 50 percent by design.")
    lines.append("")
    for model, sub in df.groupby("model"):
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| phenomenon | language | n | accuracy (%) |")
        lines.append("|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['phenomenon']} | {r['language']} | {r['count']} | {r['accuracy']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--out", default="results/summary.md")
    args = ap.parse_args()
    df = pd.read_csv(args.scores)
    tab = accuracy_table(df)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(to_markdown(tab))
    overall = (
        df.groupby("model")["congruent"].mean().mul(100).round(1)
    )
    print(tab.to_string(index=False))
    print()
    print("overall accuracy (%):")
    print(overall.to_string())


if __name__ == "__main__":
    main()
