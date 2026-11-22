"""Register this project's environment variants. After `import niryo_lift.envs`, gym.make works with these ids."""
from gymnasium.envs.registration import register

# Fix 1: the cube spawns inside the arm's reachable region, at rest on the floor.
# Everything else (gripper, damping, goal height, 50 steps) stays as in the original.
register(
    id="NiryoLiftReachable-v0",
    entry_point="niryo_lift.envs.variants:NiryoLiftVariant",
    max_episode_steps=50,
    kwargs=dict(cube_xy_center=(0.0, 0.0), cube_xy_range=0.15, spawn_on_floor=True),
)

# Final task used for all RL experiments. Every change from the original is chosen with the
# scripted controller (scripts/eval_scripted.py), before any RL training:
#   - reachable, at-rest cube spawn (see NiryoLiftReachable-v0)
#   - multi_contact: mesh fingers get several contact points instead of one
#   - finger_armature=0.1: gripper opens/closes in ~3 steps instead of ~25
#   - 110 steps = 2x the scripted controller's median steps to success (55)
# Cube mass, cube damping, finger force and goal height stay as in the original.
register(
    id="NiryoLiftFixed-v0",
    entry_point="niryo_lift.envs.variants:NiryoLiftVariant",
    max_episode_steps=110,
    kwargs=dict(
        cube_xy_center=(0.0, 0.0), cube_xy_range=0.15, spawn_on_floor=True,
        multi_contact=True, finger_armature=0.1,
    ),
)
