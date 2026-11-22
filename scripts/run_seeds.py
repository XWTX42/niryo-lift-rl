"""Run several configs x several seeds, one training process per run.

Usage:
    uv run python scripts/run_seeds.py configs/niryo_lift_fixed_sac_her.yaml \
        configs/niryo_lift_fixed_sac_her_demos.yaml --seeds 0 1 2 3 4 --jobs 2

- Runs whose directory already exists are skipped, so an interrupted batch can simply be
  restarted with the same command.
- Refuses to start if either git repository has uncommitted changes (use --allow-dirty to
  override), so every run's run_info.json points to an exact commit.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import niryo_gym
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
NIRYO_GYM_DIR = Path(niryo_gym.__file__).resolve().parents[1]


def is_dirty(repo):
    out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                         capture_output=True, text=True, check=True).stdout
    return bool(out.strip())


def run_one(config_path, seed, threads):
    experiment = yaml.safe_load(Path(config_path).read_text())["experiment"]
    run_dir = PROJECT_DIR / "runs" / experiment / f"seed{seed}"
    if run_dir.exists():
        return f"skip   {experiment} seed{seed} (exists)"
    log_path = PROJECT_DIR / "runs" / experiment / f"seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Limit threads per process so parallel runs do not fight over the CPU cores.
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    start = time.time()
    with open(log_path, "w") as log:
        result = subprocess.run([sys.executable, str(PROJECT_DIR / "scripts" / "train.py"),
                                 str(config_path), "--seed", str(seed)],
                                stdout=log, stderr=subprocess.STDOUT, env=env)
    status = "done  " if result.returncode == 0 else "FAILED"
    return f"{status} {experiment} seed{seed} in {(time.time() - start) / 60:.0f} min (log: {log_path.name})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--jobs", type=int, default=1, help="runs in parallel")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads per run")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if not args.allow_dirty:
        dirty = [str(r) for r in (PROJECT_DIR, NIRYO_GYM_DIR) if is_dirty(r)]
        if dirty:
            sys.exit(f"uncommitted changes in: {', '.join(dirty)} -- commit first or pass --allow-dirty")

    jobs = [(c, s) for c in args.configs for s in args.seeds]
    print(f"{len(jobs)} runs, {args.jobs} in parallel, {args.threads} threads each")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, config, seed, args.threads) for config, seed in jobs]
        for future in as_completed(futures):        # print each run as soon as it finishes
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
