import time

import mujoco
import mujoco.viewer

from controllers.micromouse import Micromouse
from utils.camera import setup_camera

MODEL_PATH = "models/micromouse.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot = Micromouse(model, data)


def run_command(viewer, left, right, duration):

    robot.drive(left, right)

    steps = int(duration / model.opt.timestep)

    for _ in range(steps):

        if not viewer.is_running():
            return False

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

    return True


with mujoco.viewer.launch_passive(model, data) as viewer:

    # Set the initial camera view
    setup_camera(viewer)

    # Forward
    if not run_command(viewer, 1.0, 1.0, 2.0):
        raise SystemExit

    robot.stop()

    if not run_command(viewer, 0.0, 0.0, 1.0):
        raise SystemExit

    # Reverse
    if not run_command(viewer, -1.0, -1.0, 2.0):
        raise SystemExit

    robot.stop()

    if not run_command(viewer, 0.0, 0.0, 1.0):
        raise SystemExit

    # Rotate left
    if not run_command(viewer, 1.0, -1.0, 2.0):
        raise SystemExit

    robot.stop()

    if not run_command(viewer, 0.0, 0.0, 1.0):
        raise SystemExit

    # Rotate right
    if not run_command(viewer, -1.0, 1.0, 2.0):
        raise SystemExit

    robot.stop()

    print("Movement test complete.")

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)