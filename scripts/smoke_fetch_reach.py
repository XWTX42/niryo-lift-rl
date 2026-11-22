"""Smoke test: SAC + HER on FetchReach.

FetchReach only needs the gripper to reach a goal point (no grasping), so a working
training pipeline should get close to 100% success within ~20k steps.
If this fails, the bug is in our code, not in the task.

Usage:
    uv run python scripts/smoke_fetch_reach.py            # with HER
    uv run python scripts/smoke_fetch_reach.py --no-her   # without HER, for comparison
"""
import argparse
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC, HerReplayBuffer

gym.register_envs(gymnasium_robotics)


def evaluate(model, n_episodes=20):
    """Success rate of the deterministic policy on fresh episodes (separate env)."""
    env = gym.make("FetchReach-v4")
    successes = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + episode)   # seeds never used in training
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes.append(info["is_success"])         # success = at the LAST step
    return float(np.mean(successes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-her", action="store_true")
    parser.add_argument("--steps", type=int, default=5_000)
    args = parser.parse_args()

    env = gym.make("FetchReach-v4")
    her_kwargs = {}
    if not args.no_her:
        her_kwargs = dict(
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        )

    # Hyperparameters from rl-baselines3-zoo (hyperparams/sac.yml, FetchReach)
    model = SAC(
        "MultiInputPolicy",      # observation is a dict: observation / achieved_goal / desired_goal
        env,
        learning_rate=1e-3,
        batch_size=256,
        gamma=0.95,              # short horizon (50 steps): discount the far future more
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=1_000,   # first 1000 steps: random actions to fill the buffer
        seed=0,
        device="cpu",            # small MLPs: CPU is usually faster than GPU/MPS
        verbose=0,
        **her_kwargs,
    )

    print(f"HER: {not args.no_her}, success before training: {evaluate(model):.0%}")
    start = time.time()
    for chunk in range(args.steps // 1_000):
        model.learn(total_timesteps=1_000, reset_num_timesteps=False)
        print(f"  {(chunk + 1) * 1_000:>6} steps | success {evaluate(model):4.0%} | {time.time() - start:5.0f} s")


if __name__ == "__main__":
    main()
