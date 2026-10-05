import time
import numpy as np

import mujoco
import mujoco.viewer

from controllers.micromouse import Micromouse
from controllers.sensors import Sensors
from utils.camera import setup_camera

MODEL_PATH = "models/fixed_maze.xml"

# ----------------------------------------------------------
# THRESHOLDS
# ----------------------------------------------------------

LEFT_OPEN = 0.35
FRONT_BLOCKED = 0.22

TURN_ANGLE = np.deg2rad(88)

# ----------------------------------------------------------
# STATES
# ----------------------------------------------------------

FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2

state = FORWARD
turn_start_yaw = 0.0

# ----------------------------------------------------------

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot = Micromouse(model, data)
sensors = Sensors(model, data)

robot_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "robot",
)


def get_yaw():

    R = data.xmat[robot_id].reshape(3, 3)

    forward = R[:, 0]

    return np.arctan2(
        forward[1],
        forward[0]
    )


def angle_diff(a, b):

    return np.arctan2(
        np.sin(a - b),
        np.cos(a - b)
    )


# ----------------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    for _ in range(500):
        mujoco.mj_step(model, data)

    print()
    print("========== LEFT WALL FOLLOWER ==========")

    while viewer.is_running():

        mujoco.mj_step(model, data)

        values = sensors.read()

        front = values["front"]
        left = values["left"]
        right = values["right"]

        yaw = get_yaw()

        # ------------------------------------------
        # FORWARD
        # ------------------------------------------

        if state == FORWARD:

            robot.forward(1.0)

            # Opening on the left
            if left > LEFT_OPEN:

                turn_start_yaw = yaw
                state = TURN_LEFT
                print("TURN LEFT")

            # Wall ahead
            elif front < FRONT_BLOCKED:

                turn_start_yaw = yaw
                state = TURN_RIGHT
                print("TURN RIGHT")

        # ------------------------------------------
        # TURN LEFT
        # ------------------------------------------

        elif state == TURN_LEFT:

            robot.left_arc()

            turned = abs(
                angle_diff(
                    yaw,
                    turn_start_yaw
                )
            )

            if turned >= TURN_ANGLE:

                state = FORWARD
                print("FORWARD")

        # ------------------------------------------
        # TURN RIGHT
        # ------------------------------------------

        elif state == TURN_RIGHT:

            robot.right_arc()

            turned = abs(
                angle_diff(
                    yaw,
                    turn_start_yaw
                )
            )

            if turned >= TURN_ANGLE:

                state = FORWARD
                print("FORWARD")

        viewer.sync()

        time.sleep(model.opt.timestep)

    robot.stop()