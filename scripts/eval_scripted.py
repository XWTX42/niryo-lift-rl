"""Evaluate environment settings with the scripted controller: success rate and steps needed.

Usage:
    uv run python scripts/eval_scripted.py                 # all settings
    uv run python scripts/eval_scripted.py --episodes 10   # quick check
"""
import argparse

import gymnasium as gym
import numpy as np

from niryo_lift.controllers.scripted import ScriptedGrasp
from niryo_lift.envs.variants import NiryoLiftVariant

# Shared baseline: cube spawns inside the reachable region, at rest on the floor.
BASE = dict(cube_xy_center=(0.0, 0.0), spawn_on_floor=True)

# Ablation: each entry lists what it changes on top of the baseline; everything else = original.
CONFIGS = {
    "reachable spawn (baseline)": {},
    "+ multi_contact": dict(multi_contact=True),
    "+ multi_contact + armature=1": dict(multi_contact=True, finger_armature=1),
    "+ multi_contact + armature=0.1": dict(multi_contact=True, finger_armature=0.1),
    "armature=1 only (no multi_contact)": dict(finger_armature=1),
}


def run_episode(env_kwargs, seed, max_steps):
    """Run one episode. Return the step of the first success, or None."""
    env = gym.wrappers.TimeLimit(NiryoLiftVariant(**BASE, **env_kwargs), max_steps)
    controller = ScriptedGrasp()
    obs, _ = env.reset(seed=seed)
    for step in range(1, max_steps + 1):
        obs, reward, terminated, truncated, info = env.step(controller.act(obs))
        if info["is_success"]:
            return step
        if terminated or truncated:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)   # longer than 50 on purpose: we measure how many steps are needed
    args = parser.parse_args()

    print(f"{'setting':40s} | <=50 steps | <={args.max_steps} steps | median steps | p90")
    for name, env_kwargs in CONFIGS.items():
        results = [run_episode(env_kwargs, seed, args.max_steps) for seed in range(args.episodes)]
        steps = np.array([r for r in results if r is not None])
        within_50 = np.mean([r is not None and r <= 50 for r in results])
        within_max = len(steps) / args.episodes
        median = f"{np.median(steps):.0f}" if len(steps) else "-"
        p90 = f"{np.percentile(steps, 90):.0f}" if len(steps) else "-"
        print(f"{name:40s} | {within_50:10.0%} | {within_max:11.0%} | {median:>12s} | {p90:>5s}")


if __name__ == "__main__":
    main()