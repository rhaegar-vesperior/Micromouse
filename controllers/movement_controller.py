# controllers/movement_controller.py

import mujoco
import numpy as np


class MovementController:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Actuators
        self.left_motor = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "left_motor"
        )

        self.right_motor = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "right_motor"
        )

        # Robot joints
        self.x_joint = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "robot_x"
        )

        self.y_joint = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "robot_y"
        )

        self.yaw_joint = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "robot_yaw"
        )

    # ---------------- Motor Commands ---------------- #

    def set_wheel_speeds(self, left, right):
        self.data.ctrl[self.left_motor] = left
        self.data.ctrl[self.right_motor] = right

    def stop(self):
        self.set_wheel_speeds(0.0, 0.0)

    def forward(self, speed=2.0):
        self.set_wheel_speeds(speed, speed)

    def backward(self, speed=2.0):
        self.set_wheel_speeds(-speed, -speed)

    def turn_left(self, speed=1.5):
        self.set_wheel_speeds(-speed, speed)

    def turn_right(self, speed=1.5):
        self.set_wheel_speeds(speed, -speed)

    # ---------------- Robot Pose ---------------- #

    def get_pose(self):
        x = self.data.qpos[self.x_joint]
        y = self.data.qpos[self.y_joint]
        yaw = self.data.qpos[self.yaw_joint]

        return np.array([x, y, yaw])