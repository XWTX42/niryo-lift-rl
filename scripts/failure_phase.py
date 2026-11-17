"""Classify scripted-controller episodes: success, dropped mid-air, or never lifted."""
from collections import Counter

import gymnasium as gym

from niryo_lift.controllers.scripted import ScriptedGrasp
from niryo_lift.envs.variants import NiryoLiftVariant

BASE = dict(cube_xy_center=(0.0, 0.0), spawn_on_floor=True)
CONFIGS = {
    "reachable spawn (baseline)": {},
    "+ multi_contact": dict(multi_contact=True),
    "+ multi_contact + armature=1": dict(multi_contact=True, finger_armature=1),
}

LIFTED = 0.02  # cube counts as lifted only if it rose more than 2 cm above its start height


def run_episode(env_kwargs, seed, max_steps=200):
    """Return (outcome, max_rise) for one episode.

    outcome is "success", "dropped" (was lifted, then lost), or "never lifted".
    max_rise is the highest the cube got above its start height, in metres.
    """
    env = gym.wrappers.TimeLimit(NiryoLiftVariant(**BASE, **env_kwargs), max_steps)
    controller = ScriptedGrasp()
    obs, _ = env.reset(seed=seed)

    # (1) remember where the cube started, and track the highest point it reaches
    start_z = obs["achieved_goal"][2]
    max_z = start_z

    while True:
        obs, _, terminated, truncated, info = env.step(controller.act(obs))

        # (2) update the highest point after every step
        max_z = max(max_z, obs["achieved_goal"][2])

        # (3) classify the outcome by how high the cube ever got
        max_rise = max_z - start_z
        if info["is_success"]:
            return "success", max_rise
        if terminated or truncated:
            outcome = "dropped" if max_rise > LIFTED else "never lifted"
            return outcome, max_rise


for name, env_kwargs in CONFIGS.items():
    results = [run_episode(env_kwargs, seed) for seed in range(50)]
    counts = Counter(outcome for outcome, _ in results)

    dropped_rises = [rise for outcome, rise in results if outcome == "dropped"]
    detail = ""
    if dropped_rises:
        detail = f" | dropped after rising {min(dropped_rises) * 100:.0f}-{max(dropped_rises) * 100:.0f} cm"

    print(f"{name}: {dict(counts)}{detail}")