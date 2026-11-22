"""Record everything needed to trace a result back to the exact code, config and seed."""
import json
import platform
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import niryo_gym


def _git_state(repo_dir):
    """Commit hash and whether there are uncommitted changes; None if not a git repo."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo_dir), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_run_info(project_dir):
    niryo_gym_dir = Path(niryo_gym.__file__).resolve().parents[1]
    return {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ["mujoco", "gymnasium", "gymnasium-robotics", "stable-baselines3", "torch", "numpy"]},
        "git": {"niryo-lift-rl": _git_state(project_dir), "niryo-gym": _git_state(niryo_gym_dir)},
    }


def save_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2))
