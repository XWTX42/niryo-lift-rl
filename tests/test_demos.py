"""Demonstrations must load into the HER buffer with correct episode boundaries."""
import numpy as np
from stable_baselines3 import SAC, HerReplayBuffer

from niryo_lift.controllers.scripted import ScriptedGrasp
from niryo_lift.rl.demos import add_demos_to_buffer
from niryo_lift.rl.envs import make_env


def _collect(env, seed):
    obs, _ = env.reset(seed=seed)
    controller, rows = ScriptedGrasp(), []
    while True:
        action = controller.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rows.append((obs, next_obs, action, reward, terminated, truncated))
        obs = next_obs
        if terminated or truncated:
            return rows


def test_demos_fill_her_buffer_and_relabel():
    env = make_env("NiryoLiftFixed-v0")
    rows = _collect(env, seed=0) + _collect(env, seed=1)
    keys = ["observation", "achieved_goal", "desired_goal"]
    demos = {k: np.array([r[0][k] for r in rows]) for k in keys}
    demos.update({"next_" + k: np.array([r[1][k] for r in rows]) for k in keys})
    demos.update(action=np.array([r[2] for r in rows]), reward=np.array([r[3] for r in rows]),
                 terminated=np.array([r[4] for r in rows]), truncated=np.array([r[5] for r in rows]))

    model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, learning_starts=0)
    n = add_demos_to_buffer(model.replay_buffer, demos)

    assert model.replay_buffer.size() == n == 2 * env.spec.max_episode_steps
    batch = model.replay_buffer.sample(512)        # HER relabels goals while sampling
    success_fraction = (batch.rewards == 0).float().mean().item()   # batch fields are torch tensors
    assert success_fraction > 0.3, "relabelled demo samples should often be successes"
