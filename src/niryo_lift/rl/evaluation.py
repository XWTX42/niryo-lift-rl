"""Evaluation with a fixed set of episodes, so results at different checkpoints are comparable."""
import csv
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from niryo_lift.rl.envs import make_env


def evaluate_success(model, env_id, env_kwargs=None, n_episodes=50, seed_offset=10_000):
    """Run the deterministic policy for n_episodes on a separate env.

    Episode i is reset with seed `seed_offset + i`, so every call evaluates the same episodes.
    Success is read at the LAST step of each episode.
    Returns success rate and mean final distance between achieved and desired goal.
    """
    env = make_env(env_id, env_kwargs)
    successes, final_distances = [], []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + episode)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes.append(float(info["is_success"]))
        final_distances.append(float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])))
    env.close()
    return {"success_rate": float(np.mean(successes)),
            "mean_final_distance": float(np.mean(final_distances))}


class PeriodicEval(BaseCallback):
    """SB3 callback: every `every` env steps, evaluate and append one row to a CSV file."""

    FIELDS = ["timesteps", "success_rate", "mean_final_distance", "wall_time_s"]

    def __init__(self, env_id, env_kwargs, every, n_episodes, seed_offset, csv_path):
        super().__init__()
        self.env_id, self.env_kwargs = env_id, env_kwargs
        self.every, self.n_episodes, self.seed_offset = every, n_episodes, seed_offset
        self.csv_path = csv_path
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()
        self._evaluate_and_log()                      # row for the untrained policy

    def _on_step(self):
        # Called by SB3 after every environment step; returning False would stop training.
        if self.num_timesteps % self.every == 0:
            self._evaluate_and_log()
        return True

    def _evaluate_and_log(self):
        result = evaluate_success(self.model, self.env_id, self.env_kwargs,
                                  self.n_episodes, self.seed_offset)
        row = {"timesteps": self.num_timesteps, **result,
               "wall_time_s": round(time.time() - self.start_time, 1)}
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)
        # Also log to TensorBoard (written at SB3's next dump).
        self.logger.record("eval/success_rate", result["success_rate"])
        self.logger.record("eval/mean_final_distance", result["mean_final_distance"])
        print(f"{row['timesteps']:>9} steps | success {row['success_rate']:5.1%} | "
              f"final dist {row['mean_final_distance'] * 100:5.1f} cm | {row['wall_time_s']:7.0f} s")
