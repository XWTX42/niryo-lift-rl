"""Watch a trained policy: open a live viewer window, or record a GIF.

Usage:
    uv run python scripts/watch_policy.py runs/niryo_lift_fixed_sac_her_demos/seed0            # live window
    uv run python scripts/watch_policy.py runs/niryo_lift_fixed_sac_her_demos/seed0 --gif results/figures/e1b.gif
    uv run python scripts/watch_policy.py runs/niryo_lift_fixed_sac_her_demos/seed0 --gif slow.gif --speed 0.25

Live window keys (gymnasium's MuJoCo viewer): S = half speed, F = double speed,
Space = pause/resume, Right arrow = one frame forward while paused.
"""
import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import yaml
from stable_baselines3 import SAC

from niryo_lift.rl.envs import make_env

SEED_OFFSET = 20_000   # cube positions not used in training, demos (100_000+) or evaluation (10_000+)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gif", default=None, help="save a GIF here instead of opening a window")
    parser.add_argument("--speed", type=float, default=1.0, help="GIF playback speed: 0.5 = half speed")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    env_id, env_kwargs = config["env"]["id"], config["env"].get("kwargs", {})
    # SB3 needs an env when loading a model trained with HerReplayBuffer; a plain one is enough.
    model = SAC.load(run_dir / "model.zip", env=make_env(env_id, env_kwargs), device="cpu")

    # A separate env that renders: a window ("human") or image frames ("rgb_array").
    render_mode = "rgb_array" if args.gif else "human"
    env = make_env(env_id, {**env_kwargs, "render_mode": render_mode})

    frames = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=SEED_OFFSET + episode)
        done, step = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done, step = terminated or truncated, step + 1
            if args.gif and step % 2 == 0:              # every 2nd step: 0.04 s per frame = real time at 25 fps
                frames.append(env.render())
            elif not args.gif:
                env.render()
        distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]) * 100
        print(f"episode {episode}: success={bool(info['is_success'])}, final distance {distance:.1f} cm")
    env.close()

    if args.gif:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        # Each frame covers 2 env steps = 0.04 s, so 40 ms per frame is real time.
        iio.imwrite(args.gif, np.stack(frames), duration=40 / args.speed, loop=0)   # ms per frame
        print(f"saved {args.gif} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
