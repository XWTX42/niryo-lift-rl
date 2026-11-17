"""Task-design fixes on top of the original niryo-gym Lift environment.

Principle: every fix is a constructor argument whose default reproduces the original
behaviour. "Original" and "fixed" are then just two parameter sets of the same class,
so each fix can be ablated on its own.
"""
import numpy as np
from niryo_gym.niryo_env import MujocoNiryoLiftEnv


class NiryoLiftVariant(MujocoNiryoLiftEnv):
    """NiryoLift with configurable cube spawning and a few physics parameters.

    Args:
        cube_xy_center:  Centre (x, y) of the cube spawn region. Original: (0.3, 0.0).
        cube_xy_range:   Cube is sampled uniformly in centre ± range. Original: 0.15.
        spawn_on_floor:  True = place the cube at rest on the floor. False (original) =
                         spawn at z=0 in mid-air and let it fall.
        finger_armature: Armature (extra inertia) of the finger joints. None = keep (original 10).
        cube_damping:    Damping of the cube's free joint. None = keep (original 0.1).
        cube_mass:       Cube mass in kg. None = keep (original 0.005).
        finger_force:    Max force of each finger actuator in N. None = keep (original 1).
        multi_contact:   True = enable MuJoCo's multiccd flag so convex-convex collisions
                         return several contact points. Original: False (one point per pair).
    """

    def __init__(self, cube_xy_center=(0.3, 0.0), cube_xy_range=0.15, spawn_on_floor=False,
                 finger_armature=None, cube_damping=None, cube_mass=None, finger_force=None,
                 multi_contact=False, **kwargs):
        # Set before super().__init__: the parent constructor may already need them.
        self.cube_xy_center = np.asarray(cube_xy_center, dtype=float)
        self.cube_xy_range = cube_xy_range
        self.spawn_on_floor = spawn_on_floor
        super().__init__(**kwargs)

        # Physics parameters: edit the loaded model's arrays directly (the XML stays untouched).
        if finger_armature is not None:
            for name in ["robot0:l_gripper_finger_joint", "robot0:r_gripper_finger_joint"]:
                dof = self.model.joint(name).dofadr[0]       # armature is stored per DoF, not per joint
                self.model.dof_armature[dof] = finger_armature
        if cube_damping is not None:
            dof = self.model.joint("object0:joint").dofadr[0]
            self.model.dof_damping[dof:dof + 6] = cube_damping   # a free joint has 6 DoFs
        if finger_force is not None:
            for name in ["robot0:l_gripper_finger_joint", "robot0:r_gripper_finger_joint"]:
                joint_id = self.model.joint(name).id
                self.model.jnt_actfrcrange[joint_id] = [-finger_force, finger_force]
        if cube_mass is not None:
            body = self.model.body("object0")
            scale = cube_mass / body.mass[0]
            body.mass[0] = cube_mass
            body.inertia[:] *= scale                 # same shape -> inertia scales with mass
        if multi_contact:
            self.model.opt.enableflags |= self._mujoco.mjtEnableBit.mjENBL_MULTICCD
        # Recompute MuJoCo's derived constants that depend on the edited parameters.
        self._mujoco.mj_setConst(self.model, self.data)

    def _reset_sim(self):
        # ---- Same 5 lines as the original _reset_sim: restore the initial sim state ----
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # ---- Changed part: where the cube is placed ----
        # The cube has a free joint; its qpos has 7 numbers: [x, y, z, quat w, x, y, z]
        cube_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
        offset = self.np_random.uniform(-self.cube_xy_range, self.cube_xy_range, size=2)
        cube_qpos[:2] = self.cube_xy_center + offset

        if self.spawn_on_floor:
            floor_z = self.model.geom("floor0").pos[2]          # read floor height from the model, no magic number
            half_size = self.model.geom("object0").size[2]      # cube half-size along z
            cube_qpos[2] = floor_z + half_size                  # bottom face rests on the floor
        self._utils.set_joint_qpos(self.model, self.data, "object0:joint", cube_qpos)

        # mj_forward: recompute everything derived from qpos (world poses, contacts, ...)
        self._mujoco.mj_forward(self.model, self.data)
        return True