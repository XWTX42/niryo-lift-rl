"""Load saved demonstrations and add them to an SB3 replay buffer before training."""
import numpy as np


def load_demos(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def add_demos_to_buffer(replay_buffer, demos):
    """Add every demo transition to the buffer, one step at a time, in episode order.

    SB3 buffers expect a leading "number of envs" axis (here 1), so each array gets [None].
    `done` marks episode boundaries, which HerReplayBuffer needs to relabel goals within an
    episode. Episodes that end by time limit are flagged as timeouts, so the critic still
    bootstraps from the next state instead of treating it as a true terminal state.
    """
    n = len(demos["reward"])
    for i in range(n):
        obs = {key: demos[key][i][None] for key in ["observation", "achieved_goal", "desired_goal"]}
        next_obs = {key: demos["next_" + key][i][None]
                    for key in ["observation", "achieved_goal", "desired_goal"]}
        terminated, truncated = bool(demos["terminated"][i]), bool(demos["truncated"][i])
        replay_buffer.add(
            obs, next_obs,
            action=demos["action"][i][None],
            reward=np.array([demos["reward"][i]]),
            done=np.array([terminated or truncated]),
            infos=[{"TimeLimit.truncated": truncated and not terminated}],
        )
    return n
