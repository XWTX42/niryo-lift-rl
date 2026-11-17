"""register this env, make sure import niryo_lift.envs is avaliable"""
from gymnasium.envs.registration import register

register(
    id="NiryoLiftReachable-v0",
    entry_point="niryo_lift.envs.variants:NiryoLiftVariant",
    max_episode_steps=50,
    kwargs=dict(cube_xy_center=(0.0, 0.0), cube_xy_range=0.15, spawn_on_floor=True),
)