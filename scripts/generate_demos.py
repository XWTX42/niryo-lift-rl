"""Generate demonstration episodes with the scripted controller and save every transition.

Usage:
    uv run python scripts/generate_demos.py --episodes 100
Writes data/demos_<episodes>.npz. Only successful episodes are kept.
"""
import argparse
from pathlib import Path

import numpy as np

from niryo_lift.controllers.scripted import ScriptedGrasp
from niryo_lift.rl.envs import make_env

ENV_ID = "NiryoLiftFixed-v0"
SEED_OFFSET = 100_000   # demo cube positions differ from evaluation seeds (10_000+) and training


def run_episode(env, seed):
    """Return a list of transitions (dicts) for one scripted episode, and whether it succeeded."""
    controller = ScriptedGrasp()
    obs, _ = env.reset(seed=seed)
    transitions, success = [], False
    while True:
        action = controller.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        transitions.append({
            "obs": obs, "next_obs": next_obs, "action": action, "reward": reward,
            "terminated": terminated, "truncated": truncated,
        })
        success = success or bool(info["is_success"])
        obs = next_obs
        if terminated or truncated:
            return transitions, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    env = make_env(ENV_ID)
    columns = {name: [] for name in [
        "observation", "achieved_goal", "desired_goal",
        "next_observation", "next_achieved_goal", "next_desired_goal",
        "action", "reward", "terminated", "truncated", "episode"]}
    kept = 0
    for i in range(args.episodes):
        transitions, success = run_episode(env, seed=SEED_OFFSET + i)
        if not success:
            continue
        for t in transitions:
            for key in ["observation", "achieved_goal", "desired_goal"]:
                columns[key].append(t["obs"][key])
                columns["next_" + key].append(t["next_obs"][key])
            columns["action"].append(t["action"])
            columns["reward"].append(t["reward"])
            columns["terminated"].append(t["terminated"])
            columns["truncated"].append(t["truncated"])
            columns["episode"].append(kept)
        kept += 1

    out = Path("data") / f"demos_{args.episodes}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, env_id=ENV_ID, **{k: np.array(v) for k, v in columns.items()})
    print(f"kept {kept}/{args.episodes} successful episodes, "
          f"{len(columns['reward'])} transitions -> {out}")


if __name__ == "__main__":
    main()
