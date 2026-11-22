"""Plot one scripted-controller episode: the "hand-drawn" trajectory we will use as a demonstration.

Usage:
    uv run python scripts/plot_scripted_trajectory.py --seed 0
Writes results/figures/scripted_trajectory_seed<N>.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from niryo_lift.controllers.scripted import ScriptedGrasp
from niryo_lift.rl.envs import make_env

# Colours: fixed categorical order from the dataviz reference palette (light mode).
GRIPPER, CUBE, FINGER = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#ecebe8"


def record_episode(seed):
    """Run the scripted controller once and return per-step arrays."""
    env = make_env("NiryoLiftFixed-v0")
    controller = ScriptedGrasp()
    obs, _ = env.reset(seed=seed)
    log = {"grip": [], "cube": [], "finger": [], "phase": []}
    goal = obs["desired_goal"].copy()
    success_step = None
    for step in range(env.spec.max_episode_steps):
        log["grip"].append(obs["observation"][0:3].copy())
        log["cube"].append(obs["achieved_goal"].copy())
        log["finger"].append(obs["observation"][9:11].mean())
        obs, _, terminated, truncated, info = env.step(controller.act(obs))
        log["phase"].append(controller.phase.name)
        if info["is_success"] and success_step is None:
            success_step = step + 1
        if terminated or truncated:
            break
    return {k: np.array(v) for k, v in log.items()}, goal, success_step


def phase_spans(phases):
    """Turn the per-step phase list into (name, first_step, last_step) spans."""
    spans, start = [], 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            spans.append((phases[start], start, i - 1))
            start = i
    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log, goal, success_step = record_episode(args.seed)
    steps = np.arange(len(log["phase"]))
    spans = phase_spans(log["phase"])
    horizontal = np.linalg.norm(log["grip"][:, :2] - log["cube"][:, :2], axis=1) * 100

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    fig.patch.set_facecolor("#fcfcfb")

    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        for i, (_, a, b) in enumerate(spans):               # alternate light bands per phase
            if i % 2 == 0:
                ax.axvspan(a - 0.5, b + 0.5, color=BAND, lw=0, zorder=0)
        ax.grid(axis="y", color="#e2e1dc", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    # Panel 1: height of gripper and cube, with the goal height.
    ax = axes[0]
    ax.plot(steps, log["grip"][:, 2] * 100, color=GRIPPER, lw=2)
    ax.plot(steps, log["cube"][:, 2] * 100, color=CUBE, lw=2)
    ax.axhline(goal[2] * 100, color=MUTED, lw=1, ls="--")
    end = steps[-1] + 1
    # Direct labels where the two lines are far apart (early in the episode), not at the end where they overlap.
    ax.text(6, log["grip"][5, 2] * 100, "  gripper", color=INK, fontsize=9, va="center")
    ax.text(2, log["cube"][2, 2] * 100 + 1.5, "cube", color=INK, fontsize=9, va="bottom")
    ax.text(end, goal[2] * 100 + 1.5, "goal height", color=MUTED, fontsize=9, ha="right")
    # Phase names in two staggered rows above the plot so short phases do not collide.
    top = ax.get_ylim()[1]
    for i, (name, a, b) in enumerate(spans):
        ax.text((a + b) / 2, top + (1 + 4 * (i % 2)), name.lower(), ha="center", va="bottom",
                fontsize=9, color=MUTED, clip_on=False)
    if success_step is not None:
        ax.axvline(success_step, color=INK, lw=1)
        ax.text(success_step + 1, ax.get_ylim()[0] + 3, f"success at step {success_step}", fontsize=9, color=INK)
    ax.set_ylabel("height z (cm)", color=INK)

    # Panel 2: horizontal distance gripper -> cube (alignment).
    axes[1].plot(steps, horizontal, color=GRIPPER, lw=2)
    axes[1].set_ylabel("gripper–cube\nhorizontal (cm)", color=INK)

    # Panel 3: finger opening (+10 mm fully open, -10 mm fully closed).
    axes[2].plot(steps, log["finger"] * 1000, color=FINGER, lw=2)
    axes[2].text(end, log["finger"][-1] * 1000, "finger", color=INK, fontsize=9, va="center")
    axes[2].set_ylabel("finger position (mm)", color=INK)
    axes[2].set_xlabel("environment step (0.02 s each)", color=INK)

    fig.suptitle(f"Scripted grasp on NiryoLiftFixed-v0, seed {args.seed}", color=INK, x=0.07, ha="left")
    fig.tight_layout()
    out = Path("results/figures") / f"scripted_trajectory_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
