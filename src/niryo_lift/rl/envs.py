"""Create environments by id, making sure every environment family is registered first."""
import gymnasium as gym
import gymnasium_robotics

import niryo_gym  # noqa: F401  registers the original Niryo environments
import niryo_lift.envs  # noqa: F401  registers our variants (NiryoLiftFixed-v0, ...)

gym.register_envs(gymnasium_robotics)  # registers Fetch environments


def make_env(env_id, env_kwargs=None):
    return gym.make(env_id, **(env_kwargs or {}))
