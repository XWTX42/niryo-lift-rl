"""Aggregate evaluation curves over seeds: one figure and one summary table.

Usage:
    uv run python scripts/aggregate.py niryo_lift_fixed_sac_her niryo_lift_fixed_sac_her_demos \
        --labels "E1: SAC+HER" "E1b: SAC+HER+demos" --name e1_vs_e1b

Reads runs/<experiment>/seed*/eval.csv and writes
    results/figures/<name>.png   success rate and final distance vs steps (mean and 95% CI over seeds)
    results/<name>.md            table of final performance
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e8a91c"]   # fixed categorical order (dataviz palette)
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
FINAL_EVALS = 5          # final performance = mean over the last 5 evaluations of each seed
RNG = np.random.default_rng(0)


def load_experiment(experiment):
    """Return {seed: DataFrame} for every finished seed of one experiment."""
    runs = {}
    for csv_path in sorted((PROJECT_DIR / "runs" / experiment).glob("seed*/eval.csv")):
        runs[int(csv_path.parent.name.removeprefix("seed"))] = pd.read_csv(csv_path)
    if not runs:
        raise SystemExit(f"no finished runs in runs/{experiment}")
    return runs


def bootstrap_ci(values, n_resamples=2000):
    """95% confidence interval of the mean, resampling seeds with replacement."""
    values = np.asarray(values)
    if len(values) < 2:
        return values.mean(), values.mean()
    means = RNG.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def curve(runs, column):
    """Mean and CI at every evaluation step shared by all seeds."""
    steps = sorted(set.intersection(*(set(df["timesteps"]) for df in runs.values())))
    table = np.array([[df.set_index("timesteps").loc[s, column] for s in steps] for df in runs.values()])
    ci = np.array([bootstrap_ci(table[:, j]) for j in range(len(steps))])
    return np.array(steps), table.mean(axis=0), ci[:, 0], ci[:, 1]


def summary_row(label, runs):
    finals = [df["success_rate"].iloc[-FINAL_EVALS:].mean() for df in runs.values()]
    dists = [df["mean_final_distance"].iloc[-FINAL_EVALS:].mean() * 100 for df in runs.values()]
    first_90 = [df.loc[df["success_rate"] >= 0.9, "timesteps"].min() for df in runs.values()]
    reached = [s for s in first_90 if not np.isnan(s)]
    lo, hi = bootstrap_ci(finals)
    return {
        "experiment": label,
        "seeds": len(runs),
        "final success (mean, 95% CI)": f"{np.mean(finals):.0%} [{lo:.0%}, {hi:.0%}]",
        "final distance (cm)": f"{np.mean(dists):.1f}",
        "seeds reaching 90%": f"{len(reached)}/{len(runs)}",
        "steps to 90% (median)": f"{np.median(reached) / 1000:.0f}k" if reached else "-",
    }


def markdown_table(rows):
    """Render a list of dicts as a Markdown table (no extra dependency)."""
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(row[h]) for h in header) + " |" for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    labels = args.labels or args.experiments

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor(SURFACE)
    rows = []
    for i, (experiment, label) in enumerate(zip(args.experiments, labels)):
        runs = load_experiment(experiment)
        rows.append(summary_row(label, runs))
        for ax, column, scale in [(axes[0], "success_rate", 100), (axes[1], "mean_final_distance", 100)]:
            steps, mean, lo, hi = curve(runs, column)
            ax.fill_between(steps / 1000, lo * scale, hi * scale, color=COLORS[i], alpha=0.18, lw=0)
            ax.plot(steps / 1000, mean * scale, color=COLORS[i], lw=2, label=f"{label} (n={len(runs)})")

    for ax, ylabel in zip(axes, ["success rate (%)", "final cube–goal distance (cm)"]):
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("environment steps (thousands)", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.grid(axis="y", color="#e2e1dc", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("Mean over seeds; shaded band = 95% bootstrap CI", color=MUTED, fontsize=9, x=0.07, ha="left")
    fig.tight_layout()

    out_png = PROJECT_DIR / "results" / "figures" / f"{args.name}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    table = markdown_table(rows)
    (PROJECT_DIR / "results" / f"{args.name}.md").write_text(table + "\n")
    print(table)
    print(f"\nsaved {out_png.relative_to(PROJECT_DIR)} and results/{args.name}.md")


if __name__ == "__main__":
    main()
