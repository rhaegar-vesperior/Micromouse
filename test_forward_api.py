import time

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


def reset_robot():

    # Reset slide joints and yaw
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.0

    # Zero all velocities
    data.qvel[:] = 0.0

    # Stop motors
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)


def run_test(viewer, name, left, right):

    reset_robot()

    start = data.xpos[robot_id].copy()

    print()
    print("==========", name, "==========")
    print("Start:", start)

    robot.drive(left, right)

    start_time = time.time()

    while (
        viewer.is_running()
        and time.time() - start_time < 0.7
    ):

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

    robot.stop()

    end = data.xpos[robot_id].copy()

    print("End:  ", end)
    print("Displacement:", end - start)


with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    # Forward test
    run_test(
        viewer,
        "POSITIVE",
        1.0,
        1.0
    )

    time.sleep(0.5)

    # Reverse test
    run_test(
        viewer,
        "NEGATIVE",
        -1.0,
        -1.0
    )

    print()
    print("Test complete.")

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)