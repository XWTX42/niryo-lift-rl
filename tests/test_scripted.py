"""The scripted controller must reliably solve the task with the recommended physics:
this is the evidence that the task is physically feasible."""
import gymnasium as gym

from niryo_lift.controllers.scripted import ScriptedGrasp
from niryo_lift.envs.variants import NiryoLiftVariant


def test_scripted_controller_lifts_cube_with_recommended_physics():
    successes = 0
    for seed in range(10):
        env = gym.wrappers.TimeLimit(
            NiryoLiftVariant(cube_xy_center=(0.0, 0.0), spawn_on_floor=True,
                             multi_contact=True, finger_armature=0.1),
            max_episode_steps=200,
        )
        controller = ScriptedGrasp()
        obs, _ = env.reset(seed=seed)
        for _ in range(200):
            obs, _, terminated, truncated, info = env.step(controller.act(obs))
            if info["is_success"]:
                successes += 1
                break
            if terminated or truncated:
                break
    assert successes >= 9, f"only {successes}/10 successes"
