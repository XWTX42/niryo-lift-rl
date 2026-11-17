"""test every section of robitc arm"""
import platform

import gymnasium as gym
import mujoco
import numpy as np

import niryo_gym  # noqa: F401  

print("python", platform.python_version(), "| machine", platform.machine())
print("mujoco", mujoco.__version__, "| gymnasium", gym.__version__)

env = gym.make("NiryoLift-v1")           # default：control: mocap, reward: sparse, obv: goal
obs, info = env.reset(seed=0)
print("observation 维度:", {k: v.shape for k, v in obs.items()})
print("action space:", env.action_space)  # 4d：dx, dy, dz, gripper

for t in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(f"跑完 {t + 1} 步，最后 reward={reward}, is_success={info.get('is_success')}")
env.close()

try:
    import stable_baselines3 as sb3
    import torch
    print("stable-baselines3", sb3.__version__, "| torch", torch.__version__,
          "| MPS avaliable:", torch.backends.mps.is_available())
except ImportError as e:
    print("还没装 SB3/torch:", e)