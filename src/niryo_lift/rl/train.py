"""Train one agent for one seed from a config dict, writing everything into a run directory."""
import shutil
from pathlib import Path

import yaml
from stable_baselines3 import SAC, HerReplayBuffer

from niryo_lift.rl.demos import add_demos_to_buffer, load_demos
from niryo_lift.rl.envs import make_env
from niryo_lift.rl.evaluation import PeriodicEval
from niryo_lift.rl.run_info import collect_run_info, save_json

PROJECT_DIR = Path(__file__).resolve().parents[3]
ALGORITHMS = {"SAC": SAC}          # PPO can be added here later


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config, env, seed, tensorboard_dir):
    algo = config["algo"]
    algo_class = ALGORITHMS[algo["name"]]
    kwargs = dict(algo.get("kwargs", {}))
    if algo.get("her", False):
        kwargs["replay_buffer_class"] = HerReplayBuffer
        kwargs["replay_buffer_kwargs"] = algo.get("her_kwargs", {})
    # Small networks train faster on CPU; large networks with big batches benefit from a GPU.
    device = algo.get("device", "cpu")
    return algo_class("MultiInputPolicy", env, seed=seed, device=device,
                      tensorboard_log=str(tensorboard_dir), verbose=0, **kwargs)


def train(config, seed, runs_root=PROJECT_DIR / "runs", overwrite=False):
    run_dir = Path(runs_root) / config["experiment"] / f"seed{seed}"
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} exists; pass overwrite=True to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    # Save the exact config (plus seed) and the code/package state before training starts.
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**config, "seed": seed}, sort_keys=False))
    save_json(collect_run_info(PROJECT_DIR), run_dir / "run_info.json")

    env_cfg, train_cfg = config["env"], config["train"]
    env = make_env(env_cfg["id"], env_cfg.get("kwargs"))
    model = build_model(config, env, seed, run_dir / "tb")
    if "demos" in config:
        # Pre-fill the replay buffer with scripted demonstrations (they do not count as env steps).
        n = add_demos_to_buffer(model.replay_buffer, load_demos(PROJECT_DIR / config["demos"]["path"]))
        print(f"added {n} demo transitions to the replay buffer")
    callback = PeriodicEval(
        env_id=env_cfg["id"], env_kwargs=env_cfg.get("kwargs"),
        every=train_cfg["eval_every"], n_episodes=train_cfg["eval_episodes"],
        seed_offset=train_cfg.get("eval_seed_offset", 10_000),
        csv_path=run_dir / "eval.csv",
    )
    model.learn(total_timesteps=train_cfg["total_timesteps"], callback=callback, tb_log_name=config["algo"]["name"].lower())
    model.save(run_dir / "model.zip")
    return run_dir
