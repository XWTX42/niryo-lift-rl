"""The full training pipeline must run end to end and write all expected files."""
import csv

from niryo_lift.rl.train import train

TINY_CONFIG = {
    "experiment": "pytest_tiny",
    "env": {"id": "FetchReach-v4", "kwargs": {}},
    "algo": {"name": "SAC", "her": True,
             "her_kwargs": {"n_sampled_goal": 4, "goal_selection_strategy": "future"},
             "kwargs": {"learning_starts": 100, "batch_size": 32,
                        "policy_kwargs": {"net_arch": [16, 16]}}},
    "train": {"total_timesteps": 300, "eval_every": 150, "eval_episodes": 2},
}


def test_training_pipeline_writes_run_directory(tmp_path):
    run_dir = train(TINY_CONFIG, seed=0, runs_root=tmp_path)
    for name in ["config.yaml", "run_info.json", "eval.csv", "model.zip"]:
        assert (run_dir / name).exists(), f"missing {name}"
    rows = list(csv.DictReader(open(run_dir / "eval.csv")))
    assert [int(r["timesteps"]) for r in rows] == [0, 150, 300]
