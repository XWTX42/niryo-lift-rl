import gymnasium as gym
import numpy as np
import pytest

import niryo_gym  # noqa: F401  
import niryo_lift.envs  # noqa: F401  
from niryo_lift.envs.variants import NiryoLiftVariant


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_parameters_reproduce_original(seed):
    """不传任何参数时，变体必须和原版一模一样——否则对比实验就不公平。"""
    original = gym.make("NiryoLift-v1")
    variant = NiryoLiftVariant()            # 不传参数 = 全部用默认值
    obs_a, _ = original.reset(seed=seed)
    obs_b, _ = variant.reset(seed=seed)
    for key in obs_a:
        np.testing.assert_allclose(obs_a[key], obs_b[key])


def test_reachable_cube_spawns_inside_region_and_at_rest():
    env = gym.make("NiryoLiftReachable-v0")
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        cube = obs["achieved_goal"].copy()
        assert np.all(np.abs(cube[:2]) <= 0.15 + 1e-9), f"seed {seed}: 方块超出生成范围 {cube}"

        for _ in range(5):                       # 机械臂不动，看方块会不会自己动
            obs, *_ = env.step(np.zeros(4))
        moved = np.linalg.norm(obs["achieved_goal"] - cube)
        assert moved < 0.001, f"seed {seed}: 方块在 5 步内移动了 {moved * 1000:.1f} mm，没有静止生成"