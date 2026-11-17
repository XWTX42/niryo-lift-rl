"""Hand-written (non-learning) grasp controller: a 4-phase state machine.

Purpose: check whether the cube can physically be lifted in this environment, and
measure how many steps a successful lift takes.
"""
from enum import Enum

import numpy as np

# Slices of the 25-dim observation (see _get_obs in niryo_env.py)
GRIP_POS = slice(0, 3)
FINGER_QPOS = slice(9, 11)

OPEN, CLOSE = 1.0, -1.0
MAX_STEP = 0.05          # mocap moves at most 5 cm per step (the 0.05 in _set_mocap)


class Phase(Enum):
    APPROACH = 0   # open the gripper and move above the cube
    DESCEND = 1    # descend to cube height
    GRASP = 2      # close in place until the fingers stop
    LIFT = 3       # carry the cube to the goal


class ScriptedGrasp:
    def __init__(self, hover_height=0.05, grasp_x_offset=0.005, open_finger_qpos=0.007,
                 pos_tol=0.005, settle_tol=0.0001, max_speed=0.5):
        self.hover_height = hover_height        # hover this far above the cube before descending
        self.grasp_x_offset = grasp_x_offset    # finger slot spans [grip-2.4cm, grip+1.2cm]; centre the cube in it
        self.open_finger_qpos = open_finger_qpos  # finger qpos that counts as open enough (7 mm ≈ 3.5 cm gap)
        self.pos_tol = pos_tol                  # position error below this counts as arrived
        self.settle_tol = settle_tol            # finger motion per step below this counts as stopped
        self.max_speed = max_speed              # action magnitude cap: 0.5 = at most 2.5 cm/step, avoids overshoot
        self.reset()

    def reset(self):
        self.phase = Phase.APPROACH
        self.prev_finger = None
        self.phase_log = []                     # phase at every step, for later analysis

    def _move_towards(self, grip, target):
        """Proportional control: map position error to an action clipped to [-max_speed, max_speed]."""
        return np.clip((target - grip) / MAX_STEP, -self.max_speed, self.max_speed)

    def act(self, obs):
        o = obs["observation"]
        grip, finger = o[GRIP_POS], o[FINGER_QPOS].mean()
        cube, goal = obs["achieved_goal"], obs["desired_goal"]

        grasp_point = cube + np.array([self.grasp_x_offset, 0.0, 0.0])
        hover_point = grasp_point + np.array([0.0, 0.0, self.hover_height])

        # Transitions first: once a condition holds, act with the new phase in the same step.
        if self.phase == Phase.APPROACH:
            if np.linalg.norm(hover_point - grip) < self.pos_tol and finger > self.open_finger_qpos:
                self.phase = Phase.DESCEND
        if self.phase == Phase.DESCEND:
            if np.linalg.norm(grasp_point - grip) < self.pos_tol:
                self.phase = Phase.GRASP
        if self.phase == Phase.GRASP:
            settled = self.prev_finger is not None and abs(finger - self.prev_finger) < self.settle_tol
            if settled and finger < self.open_finger_qpos:   # stopped, and actually closed
                self.phase = Phase.LIFT

        # Action of the current phase
        if self.phase == Phase.APPROACH:
            move, gripper = self._move_towards(grip, hover_point), OPEN
        elif self.phase == Phase.DESCEND:
            move, gripper = self._move_towards(grip, grasp_point), OPEN
        elif self.phase == Phase.GRASP:
            move, gripper = self._move_towards(grip, grasp_point), CLOSE
        else:  # LIFT
            move, gripper = self._move_towards(grip, goal), CLOSE

        self.prev_finger = finger
        self.phase_log.append(self.phase)
        return np.concatenate([move, [gripper]])
