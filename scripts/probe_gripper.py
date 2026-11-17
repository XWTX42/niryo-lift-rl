"""test gripper: direction, move speed, distance between two fingers

print params without trainning
"""
import gymnasium as gym
import mujoco
import numpy as np

import niryo_gym  # noqa: F401


FINGER_JOINTS = ["robot0:l_gripper_finger_joint", "robot0:r_gripper_finger_joint"]
ARMATURE_VALUE = [10, 1, 0.1]
def set_finger_armature(env, value):
    model = env.unwrapped.model
    for name in FINGER_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_id = model.jnt_dofadr[joint_id]      # set armature
        model.dof_armature[dof_id] = value

env = gym.make("NiryoLift-v1")
for i in ARMATURE_VALUE:
    set_finger_armature(env=env, value=i)
    env.reset(seed=0)
    model = env.unwrapped.model   # MjModel: didn't change
    data = env.unwrapped.data     # MjData：change

    # MuJoCo use id in config to print params
    JOINT = mujoco.mjtObj.mjOBJ_JOINT
    GEOM = mujoco.mjtObj.mjOBJ_GEOM
    left_joint = mujoco.mj_name2id(model, JOINT, "robot0:l_gripper_finger_joint")
    left_geom = mujoco.mj_name2id(model, GEOM, "robot0:l_gripper_finger_link")
    right_geom = mujoco.mj_name2id(model, GEOM, "robot0:r_gripper_finger_link")
    cube_geom = mujoco.mj_name2id(model, GEOM, "object0")


    def finger_qpos():
        """the location of left finger"""
        return data.qpos[model.jnt_qposadr[left_joint]]


    def finger_gap():
        """the nearst distance between two fingers"""
        fromto = np.zeros(6)
        return mujoco.mj_geomDistance(model, data, left_geom, right_geom, 0.2, fromto)


    cube_width = 2 * model.geom_size[cube_geom][0]   # geom_size half side
    print(f"the width of cube: {cube_width * 100:.1f} cm")
    print(f"the distance of inital fingers: {finger_gap() * 100:.2f} cm\n")

    print(" commands  steps   finger location(mm)   distance(cm)")
    for cmd in [+1.0, -1.0]:
        for step in range(1, 26):
            env.step(np.array([0.0, 0.0, 0.0, cmd]))   # only change finger
            if step % 5 == 0:
                print(f" {cmd:+.0f}    {step:3d}     {finger_qpos() * 1000:+6.1f}        {finger_gap() * 100:5.2f}")

    env.close()