import time
import numpy as np

import mujoco
import mujoco.viewer

from controllers.micromouse import Micromouse
from utils.camera import setup_camera

MODEL_PATH = "models/fixed_maze.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot = Micromouse(model, data)

robot_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "robot"
)


def get_yaw():

    R = data.xmat[robot_id].reshape(3, 3)

    # Robot's local +X axis in world coordinates
    forward = R[:, 0]

    return np.arctan2(
        forward[1],
        forward[0]
    )


def reset():

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)


with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    setup_camera(viewer)

    reset()

    # Let the robot settle
    for _ in range(500):

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

    initial_yaw = get_yaw()

    print("Initial world yaw:",
          np.degrees(initial_yaw))

    # -----------------------------------------------------
    # TURN LEFT
    # -----------------------------------------------------

    robot.turn_left(1.0)

    start_time = time.time()

    while (
        viewer.is_running()
        and time.time() - start_time < 0.5
    ):

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

    robot.stop()

    final_yaw = get_yaw()

    delta = np.arctan2(
        np.sin(final_yaw - initial_yaw),
        np.cos(final_yaw - initial_yaw)
    )

    print()
    print("Final world yaw:",
          np.degrees(final_yaw))

    print(
        "Yaw change:",
        np.degrees(delta),
        "degrees"
    )

    print()
    print("Expected approximately:")
    print("90 degrees for a left turn")

    print()
    print("Close the viewer window to exit.")

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)

print("Turn test finished.")